import sys
import getpass
import requests

VGG_FACE_URL = "http://zeus.robots.ox.ac.uk/vgg_face2/login/"
IMAGE_URL = "http://zeus.robots.ox.ac.uk/vgg_face2/get_file?fname=vggface2_train.tar.gz"
TEST_IMAGE_URL="http://zeus.robots.ox.ac.uk/vgg_face2/get_file?fname=vggface2_test.tar.gz"

print('Please enter your VGG Face 2 credentials:')
user_string = input('    User: ')
password_string = getpass.getpass(prompt='    Password: ')

credential = {
    'username': user_string,
    'password': password_string
}

session = requests.session()
r = session.get(VGG_FACE_URL)

if 'csrftoken' in session.cookies:
    csrftoken = session.cookies['csrftoken']
elif 'csrf' in session.cookies:
    csrftoken = session.cookies['csrf']
else:
    raise ValueError("Unable to locate CSRF token.")

credential['csrfmiddlewaretoken'] = csrftoken

r = session.post(VGG_FACE_URL, data=credential)

imagefiles = IMAGE_URL.split('=')[-1]

with open(imagefiles, "wb") as files:
    print(f"Downloading the file: `{imagefiles}`")
    r = session.get(IMAGE_URL, data=credential, stream=True)
    bytes_written = 0
    for data in r.iter_content(chunk_size=400096):
        files.write(data)
        bytes_written += len(data)
        MegaBytes = bytes_written / (1024 * 1024)
        sys.stdout.write(f"\r{MegaBytes:0.2f} MiB downloaded...")
        sys.stdout.flush()

print("\n Images are successfully downloaded. Exiting the process.")