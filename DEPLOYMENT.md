# Render deployment

The project uses Python 3.10 (`.python-version`) because TensorFlow 2.10 supports Python 3.10 but not current Python releases.

`render.yaml` configures the Render build and start commands. It uses `render-requirements.txt`, a UTF-8 dependency file. The existing `requirements.txt` is UTF-16 encoded, which `pip` on Render cannot read.

## Push the model with Git LFS

The trained file `static/models/brain_tumor_vgg16_90acc.h5` is about 193 MB. GitHub rejects normal files over 100 MB, so it must be stored using Git LFS:

```powershell
git lfs install
git add .gitattributes
git add static/models/brain_tumor_vgg16_90acc.h5
git add .
git commit -m "Prepare Render deployment"
git push origin main
```

Then create a Render **Web Service** from this repository. Render detects `render.yaml`; if configuring it manually, use `pip install -r render-requirements.txt` as the build command and `gunicorn --workers 1 --threads 2 --timeout 180 app:app` as the start command.

Use at least 2 GB RAM. TensorFlow plus VGG16 is not suitable for a 512 MB instance. Uploaded images use Render's ephemeral filesystem and are cleared on redeploy or restart.
