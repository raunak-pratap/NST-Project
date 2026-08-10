# NST-Project

Neural Style Transfer (NST) web application — a Flask frontend that runs an AdaIN-based style transfer model implemented in PyTorch. Users can upload a content image and a style image, adjust an alpha blending parameter, and get a stylized output image.

Status
------
Prototype / demo — working Flask app with example models and a simple UI.

Highlights
---------
- Flask web UI (app.py) with file uploads and an alpha slider
- PyTorch AdaIN implementation (encoder + decoder) for fast neural style transfer
- Example model weights included: `vgg_normalised.pth` (encoder) and a decoder checkpoint under `experiment/final_exp/decoder_10.pth` (if present)
- Procfile + gunicorn provided for process-based deployment
- Image resizing to 256x256 in the server to limit RAM usage (helps avoid OOM on small hosts)

Tech stack
----------
- Python 3.8+
- Flask
- PyTorch & torchvision
- WTForms / Flask-WTF for upload forms
- Flask-Bootstrap for simple styling
- Gunicorn (for production web server)

Repository layout
-----------------
- app.py — Flask application and HTTP routes (entrypoint)
- requirements.txt — Python dependencies (includes a PyTorch CPU wheel index by default)
- Procfile — `web: gunicorn --bind :$PORT app:app`
- templates/ — Jinja2 templates (index.html UI)
- static/ — static assets and `static/uploads/` for uploaded/generated images
- examples/ — example content/style images served by the app
- experiment/final_exp/ — directory where decoder checkpoints may be stored
- vgg_normalised.pth — encoder weights used by the model
- train.py — training / experiment utilities (if you want to fine-tune or train new decoders)
- utils/ — model and helper code (VGGEncoder, Decoder, AdaIN utilities)

Quickstart (local)
------------------
1. Clone the repo and enter the directory:

   git clone https://github.com/raunak-pratap/NST-Project.git
   cd NST-Project

2. Create and activate a virtual environment (recommended):

   python -m venv .venv
   # macOS / Linux
   source .venv/bin/activate
   # Windows (PowerShell)
   .\.venv\Scripts\Activate.ps1

3. Install dependencies:

   pip install -r requirements.txt

   Note: The included `requirements.txt` uses an extra-index-url pointing to a PyTorch CPU wheel. If you have a CUDA-capable GPU and want GPU support, install the appropriate torch and torchvision wheels from https://pytorch.org/ (choose the correct CUDA version) instead of the CPU-only wheels.

4. (Optional) If model weights are not present in the repository (or are tracked with Git LFS), download them and place them in the expected paths:

   - Encoder: `vgg_normalised.pth` (root)
   - Decoder checkpoint: `experiment/final_exp/decoder_10.pth`

   These files can be large. For deployments (Heroku, Render, etc.), consider hosting them on external storage (S3, Google Cloud Storage) and add a small download-on-start script, or use Git LFS.

5. Run the app locally (development/demo):

   python app.py

   The app listens on port 8080 by default. You can change the port using the PORT environment variable.

Usage
-----
- Open the root URL (http://localhost:8080 by default).
- Upload a content image and a style image (png/jpg/jpeg).
- Optionally adjust the Alpha slider (0.0 = content, 1.0 = full stylization) and press "Transfer Style".
- The stylized result is saved to `static/uploads/` and served back to the browser.

Environment variables
---------------------
- SECRET_KEY — Flask secret key (defaults to a fallback value if not set)
- PORT — server port (defaults to 8080)

Production & Deployment notes
----------------------------
- The included `Procfile` is configured to run gunicorn: `web: gunicorn --bind :$PORT app:app`.
- Model weights can be large and increase slug size for PaaS platforms like Heroku. If you plan to deploy to Heroku or similar, consider:
  - Storing large model files in external object storage and downloading them at boot time.
  - Using Git LFS for model files.
  - Serving the model from a separate API service (if you need GPU-backed inference).
- The server resizes images to 256x256 by default to reduce memory usage; if you have more resources you can increase this.

Performance & GPU
-----------------
- The app will use CUDA if available (PyTorch's device detection). For GPU inference, install the CUDA-enabled torch wheels.
- On CPU-only hosts, inference will be slower; the code uses torch.no_grad() and resizes inputs to conserve RAM.

Testing
-------
- There are no formal tests in the repository yet. You can run basic end-to-end checks by starting the server and trying the provided images in `examples/`.

Adding or updating model weights
-------------------------------
If you train new decoder checkpoints with `train.py`, save them under `experiment/final_exp/` and ensure the checkpoint filename matches what `app.py` expects (e.g., `decoder_10.pth`) or update `app.py` to load your preferred path.

Security & File uploads
-----------------------
- Uploaded filenames are sanitized with `werkzeug.utils.secure_filename` and saved under `static/uploads/`.
- Allowed file extensions: png, jpg, jpeg.
- For public deployments, consider adding stricter file size limits and scanning uploaded content.

Contributing
------------
Contributions are welcome. Open an issue to discuss major changes, then submit a pull request with a clear description and tests where appropriate.

License
-------
No license file is included. Add a LICENSE file to make your intended license explicit (for example MIT, Apache-2.0, etc.). See https://choosealicense.com/ for help.

Contact
-------
Maintainer: raunak-pratap (https://github.com/raunak-pratap)


---

If you'd like, I can also:
- Add a small `download_models.py` helper that fetches weights from a provided URL at startup.
- Add a sample `.env` and `.env.sample` showing SECRET_KEY and PORT usage.
- Add Git LFS instructions or a .gitattributes entry for model files.

Tell me which of the above you'd like and I'll add the files and update the README accordingly.
