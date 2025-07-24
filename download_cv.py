from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.credentials import Credentials

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

creds = Credentials.from_authorized_user_file('token.json', SCOPES)

service = build('drive', 'v3', credentials=creds)

folder_id = "1nrkXMY3C1sX1t6mURMOvRYJW0JpOo_3Z"

results = service.files().list(
    q=f"'{folder_id}' in parents", fields = 'files(id, name)'
).execute()

files = results.get('files', [])

if not files:
    raise FileNotFoundError('Not Files in Results.')
else:
    for file in files:
        requests = service.files().get_media(fileId=file['id'])
        file_path = f"./curriculum/{file['name']}"
        with open(file_path, 'wb') as file:
            downloader = MediaIoBaseDownload(file, requests)
            done = False
            while not done:
                status, done = downloader.next_chunk()