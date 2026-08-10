import os
import gc
import uuid
import traceback
import torch
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image
from torchvision import transforms

# Import your existing AdaIN code
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization, calc_mean_std

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'supersecretkey')

# Use absolute paths to prevent static directory lookup issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField('Alpha', default=1.0)
    submit = SubmitField('Transfer Style')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Paths
decoder_path = os.path.join(BASE_DIR, "experiment", "final_exp", "decoder_10.pth")
encoder_path = os.path.join(BASE_DIR, "vgg_normalised.pth")

# Load Models
encoder = VGGEncoder(encoder_path).to(device)
decoder = Decoder().to(device)

if os.path.exists(decoder_path):
    state_dict = torch.load(decoder_path, map_location=device)
    fixed_state_dict = {}
    for k, v in state_dict.items():
        fixed_state_dict[k.replace('decoder', 'net')] = v
    decoder.load_state_dict(fixed_state_dict)

encoder.eval()
decoder.eval()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

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

    # ONLY check for validation and errors on POST submit
    if request.method == 'POST':
        # Handle uploaded content file or hidden field fallback
        if form.content.data and hasattr(form.content.data, 'filename') and form.content.data.filename:
            if allowed_file(form.content.data.filename):
                content_filename = secure_filename(form.content.data.filename)
                form.content.data.save(os.path.join(app.config['UPLOAD_FOLDER'], content_filename))
                form.content_path.data = content_filename
        else:
            content_filename = form.content_path.data

        # Handle uploaded style file or hidden field fallback
        if form.style.data and hasattr(form.style.data, 'filename') and form.style.data.filename:
            if allowed_file(form.style.data.filename):
                style_filename = secure_filename(form.style.data.filename)
                form.style.data.save(os.path.join(app.config['UPLOAD_FOLDER'], style_filename))
                form.style_path.data = style_filename
        else:
            style_filename = form.style_path.data

        if content_filename and style_filename:
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
            
            try:
                content_image = Image.open(content_path).convert("RGB")
                style_image = Image.open(style_path).convert("RGB")

                alpha = float(form.alpha.data) if form.alpha.data is not None else 1.0

                stylized_tensor = style_transfer(
                    content_image,
                    style_image,
                    encoder,
                    decoder,
                    alpha,
                    device
                )

                # Generate unique filename to avoid browser image caching collisions
                uid = uuid.uuid4().hex[:6]
                result_filename = f"stylized_{uid}_{content_filename}"
                result_path = os.path.join(app.config["UPLOAD_FOLDER"], result_filename)
                save_image(stylized_tensor, result_path)

                result_image = result_filename

                # Free Python RAM after model completion
                gc.collect()

            except Exception as e:
                traceback.print_exc()
                error = f"Error during style transfer processing: {str(e)}"
        else:
            if not content_filename:
                error = 'Please upload a content image.'
            elif not style_filename:
                error = 'Please upload a style image.'

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
    return 'OK', 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
