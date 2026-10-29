import os
import tempfile
import unittest
from io import BytesIO

from PIL import Image
from werkzeug.datastructures import FileStorage

import app as application


def image_upload(filename='image.png'):
    stream = BytesIO()
    Image.new('RGB', (4, 4), 'white').save(stream, format='PNG')
    stream.seek(0)
    return FileStorage(stream=stream, filename=filename, content_type='image/png')


class ApplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        application.app.config.update(TESTING=True, WTF_CSRF_ENABLED=False)

    def test_bundled_model_loads_and_healthcheck_is_ready(self):
        self.assertIsNone(application.model_error)
        response = application.app.test_client().get('/health')
        self.assertEqual(response.status_code, 200)

    def test_uploads_are_verified_and_given_unique_names(self):
        with tempfile.TemporaryDirectory() as directory:
            original_folder = application.app.config['UPLOAD_FOLDER']
            application.app.config['UPLOAD_FOLDER'] = directory
            try:
                first = application.save_upload(image_upload('same-name.png'))
                second = application.save_upload(image_upload('same-name.png'))
            finally:
                application.app.config['UPLOAD_FOLDER'] = original_folder

            self.assertNotEqual(first, second)
            self.assertTrue(first.endswith('_same-name.png'))
            self.assertTrue(os.path.isfile(os.path.join(directory, first)))
            self.assertTrue(os.path.isfile(os.path.join(directory, second)))

    def test_invalid_image_is_rejected(self):
        invalid = FileStorage(stream=BytesIO(b'not an image'), filename='image.png')
        with self.assertRaisesRegex(ValueError, 'not a valid image'):
            application.save_upload(invalid)

    def test_form_rejects_an_out_of_range_alpha(self):
        client = application.app.test_client()
        response = client.post('/', data={
            'content': (BytesIO(b'not reached'), 'content.png'),
            'style': (BytesIO(b'not reached'), 'style.png'),
            'alpha': '1.5',
        }, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Number must be between 0 and 1.', response.data)


if __name__ == '__main__':
    unittest.main()
