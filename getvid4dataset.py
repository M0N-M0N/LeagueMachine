import os
import yt_dlp as youtube_dl

def download_youtube_playlist(playlist_url, output_folder):
    #set download option
    ydl_opts = {
        'format': 'bestvideo/best',
        'outtmpl': os.path.join(output_folder, '%(title)s.%(ext)s'),
        'writethumbnail': True,
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([playlist_url])

    return True

if __name__ == "__main__":
    playlist_url = "https://www.youtube.com/playlist?list=PLwg1qJ9kcSUKlkGYLDbw_IQPvub8rV4U9"
    output_folder = "datasetvids/"

    download_youtube_playlist(playlist_url, output_folder)