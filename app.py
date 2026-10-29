import os
import gc
import uuid
import traceback
import time
from threading import Lock
import torch
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask_wtf.file import FileAllowed, FileRequired
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField
from wtforms.validators import NumberRange
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

# Import your existing AdaIN code
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization, calc_mean_std

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-me-in-production')
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_UPLOAD_BYTES', 10 * 1024 * 1024))

# Use absolute paths to prevent static directory lookup issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class UploadForm(FlaskForm):
    content = FileField('Content Image', validators=[
        FileRequired('Choose a content image.'),
        FileAllowed({'png', 'jpg', 'jpeg'}, 'Use a PNG or JPEG image.'),
    ])
    style = FileField('Style Image', validators=[
        FileRequired('Choose a style image.'),
        FileAllowed({'png', 'jpg', 'jpeg'}, 'Use a PNG or JPEG image.'),
    ])
    alpha = FloatField('Alpha', default=1.0, validators=[NumberRange(0, 1)])
    submit = SubmitField('Transfer Style')

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# Model Paths
decoder_path = os.path.join(BASE_DIR, "experiment", "final_exp", "decoder_10.pth")
encoder_path = os.path.join(BASE_DIR, "vgg_normalised.pth")

def load_models():
    """Load only validated model weights; never silently use a random decoder."""
    if not os.path.isfile(encoder_path):
        raise FileNotFoundError(f'Encoder weights were not found: {encoder_path}')
    if not os.path.isfile(decoder_path):
        raise FileNotFoundError(f'Decoder weights were not found: {decoder_path}')

    loaded_encoder = VGGEncoder(encoder_path).to(device)
    loaded_decoder = Decoder().to(device)
    checkpoint = torch.load(decoder_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get('state_dict', checkpoint)
    normalized_state_dict = {}
    for key, value in state_dict.items():
        key = key.removeprefix('module.').removeprefix('decoder.')
        # Older training checkpoints contain Sequential keys such as "1.weight";
        # Decoder stores that Sequential module under ``net``.
        if not key.startswith('net.'):
            key = f'net.{key}'
        normalized_state_dict[key] = value
    loaded_decoder.load_state_dict(normalized_state_dict, strict=True)
    loaded_encoder.eval()
    loaded_decoder.eval()
    return loaded_encoder, loaded_decoder


encoder = decoder = None
model_error = None
if not os.environ.get('NST_SKIP_MODEL_LOAD'):
    try:
        encoder, decoder = load_models()
    except Exception as exc:
        model_error = exc
        app.logger.exception('Unable to load the style-transfer model')

# Limit simultaneous inference requests: the model is memory intensive and a
# queue is preferable to exhausting a small deployment and returning a 502.
inference_lock = Lock()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def save_upload(upload):
    """Validate image bytes and store them under an unguessable, unique name."""
    if not upload or not upload.filename or not allowed_file(upload.filename):
        raise ValueError('Use a PNG or JPEG image.')

    try:
        Image.open(upload.stream).verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError('The uploaded file is not a valid image.') from exc
    finally:
        upload.stream.seek(0)

    filename = f'{uuid.uuid4().hex}_{secure_filename(upload.filename)}'
    upload.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return filename


def remove_expired_uploads():
    """Bound disk use without touching files that predate the UUID naming scheme."""
    retention_seconds = int(os.environ.get('UPLOAD_RETENTION_SECONDS', 24 * 60 * 60))
    now = time.time()
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        is_managed_upload = len(filename) > 33 and filename[32] == '_'
        is_managed_result = filename.startswith('stylized_') and len(filename) > 42 and filename[41] == '_'
        if is_managed_upload or is_managed_result:
            try:
                if os.path.isfile(path) and now - os.path.getmtime(path) > retention_seconds:
                    os.remove(path)
            except OSError:
                app.logger.warning('Could not remove expired upload: %s', filename)

def style_transfer(content_image, style_image, encoder, decoder, alpha, device):
    # Resize images to (256, 256) to fit within Render's 512MB RAM limit and avoid 502 OOM crashes
    transform_pipeline = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    content_tensor = transform_pipeline(content_image).unsqueeze(0).to(device)
    style_tensor = transform_pipeline(style_image).unsqueeze(0).to(device)

    # Disable gradient tracking to conserve RAM
    with torch.no_grad():
        content_feats = encoder(content_tensor, is_test=True)
        style_feats = encoder(style_tensor, is_test=True)

        stylized_feats = adaptive_instance_normalization(content_feats, style_feats)
        stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats

        stylized_image = decoder(stylized_feats)

    return stylized_image

def save_image(image, path):
    image = image.cpu().clone().squeeze(0).clamp(0, 1)
    pil_image = transforms.ToPILImage()(image)
    pil_image.save(path)

@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if form.validate_on_submit():
        if model_error or encoder is None or decoder is None:
            error = 'The style-transfer model is currently unavailable. Please try again later.'
        else:
            try:
                remove_expired_uploads()
                content_filename = save_upload(form.content.data)
                style_filename = save_upload(form.style.data)
                content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
                style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
                content_image = Image.open(content_path).convert("RGB")
                style_image = Image.open(style_path).convert("RGB")

                with inference_lock:
                    stylized_tensor = style_transfer(
                        content_image, style_image, encoder, decoder, form.alpha.data, device
                    )

                # Generate unique filename to avoid browser image caching collisions
                result_filename = f"stylized_{uuid.uuid4().hex}_{content_filename}"
                result_path = os.path.join(app.config["UPLOAD_FOLDER"], result_filename)
                save_image(stylized_tensor, result_path)

                result_image = result_filename

                # Free Python RAM after model completion
                gc.collect()

            except (UnidentifiedImageError, OSError, ValueError) as exc:
                app.logger.info('Rejected invalid upload: %s', exc)
                error = str(exc)
            except Exception:
                traceback.print_exc()
                error = 'Style transfer could not be completed. Please try different images.'
    elif request.method == 'POST':
        error = next((message for messages in form.errors.values() for message in messages),
                     'Please correct the form and try again.')

    return render_template(
        'index.html',
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error
    )

@app.route('/uploads/<path:filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'examples'), filename)

# Health check endpoint for load balancers / PaaS health probes
@app.route('/health', methods=['GET'])
def health():
    if model_error or encoder is None or decoder is None:
        return 'Model unavailable', 503
    return 'OK', 200


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(_error):
    megabytes = app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
    return render_template('index.html', form=UploadForm(), result_image=None,
                           content_image=None, style_image=None,
                           error=f'Each upload must be {megabytes:g} MB or smaller.'), 413

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
