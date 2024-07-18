from concurrent.futures import process
import os
import shutil
from huggingface_hub import snapshot_download
import gradio as gr
from gradio_client import Client, handle_file
from mutagen.mp3 import MP3
from pydub import AudioSegment
from PIL import Image
import ffmpeg
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from scripts.inference import inference_process

#from /content/drive/MyDrive/Project_584/scripts/inference.py 
import argparse
import uuid

is_shared_ui = True 

hallo_dir = snapshot_download(repo_id="fudan-generative-ai/hallo", local_dir="pretrained_models")

AUDIO_MAX_DURATION = 40000

#############
# UTILITIES #
#############

def is_mp3(file_path):
    try:
        audio = MP3(file_path)
        return True
    except Exception as e:
        return False

def convert_mp3_to_wav(mp3_file_path, wav_file_path):
    # Load the MP3 file
    audio = AudioSegment.from_mp3(mp3_file_path)
    # Export as WAV file
    audio.export(wav_file_path, format="wav")
    return wav_file_path

    
def trim_audio(file_path, output_path, max_duration):
    # Load the audio file
    audio = AudioSegment.from_wav(file_path)
    
    # Check the length of the audio in milliseconds
    audio_length = len(audio)
    
    # If the audio is longer than the maximum duration, trim it
    if audio_length > max_duration:
        trimmed_audio = audio[:max_duration]
    else:
        trimmed_audio = audio
    
    # Export the trimmed audio to a new file
    trimmed_audio.export(output_path, format="wav")

    return output_path


def add_silence_to_wav(wav_file_path, duration_s=1):
    # Load the WAV file
    audio = AudioSegment.from_wav(wav_file_path)
    # Create 1 second of silence
    silence = AudioSegment.silent(duration=duration_s * 1000)  # duration is in milliseconds
    # Add silence to the end of the audio file
    audio_with_silence = audio + silence
    # Export the modified audio
    audio_with_silence.export(wav_file_path, format="wav")
    return wav_file_path

def check_mp3(file_path):
    
    if is_mp3(file_path):
        unique_id = uuid.uuid4()
        wav_file_path = f"{os.path.splitext(file_path)[0]}-{unique_id}.wav"
        converted_audio = convert_mp3_to_wav(file_path, wav_file_path)
        print(f"File converted to {wav_file_path}")
        
        return converted_audio, gr.update(value=converted_audio, visible=True)
    else:
        print("The file is not an MP3 file.")
        
        return file_path, gr.update(value=file_path, visible=True)

def check_and_convert_webp_to_png(input_path, output_path):
    try:
        # Open the image file
        with Image.open(input_path) as img:
            # Check if the image is in WebP format
            if img.format == 'WEBP':
                # Convert and save as PNG
                img.save(output_path, 'PNG')
                print(f"Converted {input_path} to {output_path}")
                return output_path
            else:
                print(f"The file {input_path} is not in WebP format.")
                return input_path
    except IOError:
        print(f"Cannot open {input_path}. The file might not exist or is not an image.")

def convert_user_uploded_webp(input_path):

    # convert to png if necessary
    input_file = input_path
    unique_id = uuid.uuid4()
    output_file = f"converted_to_png_portrait-{unique_id}.png"
    ready_png = check_and_convert_webp_to_png(input_file, output_file)
    print(f"PORTRAIT PNG FILE: {ready_png}")
    return ready_png

def clear_audio_elms():
    return gr.update(value=None, visible=False)

def change_video_codec(input_file, output_file, codec='libx264', audio_codec='aac'):
    try:
        (
            ffmpeg
            .input(input_file)
            .output(output_file, vcodec=codec, acodec=audio_codec)
            .run(overwrite_output=True)
        )
        print(f'Successfully changed codec of {input_file} and saved as {output_file}')
    except ffmpeg.Error as e:
        print(f'Error occurred: {e.stderr.decode()}')


#######################################################
# Gradio APIs for optional image and voice generation #
#######################################################

def generate_portrait(prompt_image):
    if prompt_image is None or prompt_image == "":
        raise gr.Error("Can't generate a portrait without a prompt !")
    
    try:
        client = Client("ByteDance/SDXL-Lightning")
    except:
        raise gr.Error(f"ByteDance/SDXL-Lightning space's api might not be ready, please wait, or upload an image instead.")
    
    result = client.predict(
        prompt = prompt_image,
        ckpt = "4-Step",
        api_name = "/generate_image"
    )
    print(result)

    # convert to png if necessary
    input_file = result
    unique_id = uuid.uuid4()
    output_file = f"converted_to_png_portrait-{unique_id}.png"
    ready_png = check_and_convert_webp_to_png(input_file, output_file)
    print(f"PORTRAIT PNG FILE: {ready_png}")

    return ready_png

def generate_voice_with_parler(prompt_audio, voice_description):
    if prompt_audio is None or prompt_audio == "" :
        raise gr.Error(f"Can't generate a voice without text to synthetize !")
    if voice_description is None or voice_description == "":
        gr.Info(
            "For better control, You may want to provide a voice character description next time.",
            duration = 10,
            visible = True
        )
    try:
        client = Client("parler-tts/parler_tts_mini")
    except:
        raise gr.Error(f"parler-tts/parler_tts_mini space's api might not be ready, please wait, or upload an audio instead.")
    
    result = client.predict(
        text = prompt_audio,
        description = voice_description,
        api_name = "/gen_tts"
    )
    print(result)
    return result, gr.update(value=result, visible=True)

def get_whisperspeech(prompt_audio_whisperspeech, audio_to_clone):
    try:
        client = Client("collabora/WhisperSpeech")
    except:
        raise gr.Error(f"collabora/WhisperSpeech space's api might not be ready, please wait, or upload an audio instead.")
    
    result = client.predict(
        multilingual_text = prompt_audio_whisperspeech,
        speaker_audio = handle_file(audio_to_clone),
        speaker_url = "",
        cps = 14,
        api_name = "/whisper_speech_demo"
    )
    print(result) 
    return result, gr.update(value=result, visible=True)


########################
# TALKING PORTRAIT GEN #
########################

def run_hallo(source_image, driving_audio, progress=gr.Progress(track_tqdm=True)):

    unique_id = uuid.uuid4()
    
    args = argparse.Namespace(
        config = 'configs/inference/default.yaml',
        source_image = source_image,
        driving_audio = driving_audio,
        output = f'output-{unique_id}.mp4',
        pose_weight = 1.0,
        face_weight = 1.0,
        lip_weight = 1.0,
        face_expand_ratio = 1.2,
        checkpoint = None
    )
    
    inference_process(args)
    return f'output-{unique_id}.mp4' 

def generate_talking_portrait(portrait, voice, progress=gr.Progress(track_tqdm=True)):

    if portrait is None: 
        raise gr.Error("Please provide a portrait to animate.")
    
    if voice is None:
        raise gr.Error("Please provide audio (4 seconds max).")
    
    if is_shared_ui :
        # Trim audio to AUDIO_MAX_DURATION for better shared experience with community
        input_file = voice
        unique_id = uuid.uuid4()
        trimmed_output_file = f"-{unique_id}.wav"
        trimmed_output_file = trim_audio(input_file, trimmed_output_file, AUDIO_MAX_DURATION)
        voice = trimmed_output_file

    # Add 1 second of silence at the end to avoid last word being cut by hallo
    ready_audio = add_silence_to_wav(voice)
    print(f"1 second of silence added to {voice}")

    # Call hallo 
    talking_portrait_vid = run_hallo(portrait, ready_audio)

    # Convert video to readable format
    
    final_output_file = f"converted_{talking_portrait_vid}"
    change_video_codec(talking_portrait_vid, final_output_file)
    
    return final_output_file


css = '''
#col-container {
    margin: 0 auto;
}
#column-names {
    margin-top: 50px;
}
#main-group {
    background-color: none;
}
.tabs {
    background-color: unset;
}
#image-block {
    flex: 1;
}
#video-block {
    flex: 9;
}
#audio-block, #audio-clone-elm {
    flex: 1;
}
div#audio-clone-elm > .audio-container > button {
    height: 180px!important;
}
div#audio-clone-elm > .audio-container > button > .wrap {
    font-size: 0.9em;
}
#text-synth, #voice-desc{
    height: 130px;
}
#text-synth-wsp {
    height: 120px;
}
#audio-column, #result-column {
    display: flex;
}
#gen-voice-btn {
    flex: 1;
}
#parler-tab, #whisperspeech-tab {
    padding: 0;
}
#main-submit{
    flex: 1;
}
#pro-tips {
    margin-top: 50px;
}
div#warning-ready {
    background-color: #ecfdf5;
    padding: 0 16px 16px;
    margin: 20px 0;
    color: #030303!important;
}
div#warning-ready > .gr-prose > h2, div#warning-ready > .gr-prose > p {
    color: #057857!important;
}
div#warning-duplicate {
    background-color: #ebf5ff;
    padding: 0 16px 16px;
    margin: 20px 0;
    color: #030303!important;
}
div#warning-duplicate > .gr-prose > h2, div#warning-duplicate > .gr-prose > p {
    color: #0f4592!important;
}
div#warning-duplicate strong {
    color: #0f4592;
}
p.actions {
    display: flex;
    align-items: center;
    margin: 20px 0;
}
div#warning-duplicate .actions a {
    display: inline-block;
    margin-right: 10px;
}
.dark #warning-duplicate {
    background-color: #0c0c0c !important;
    border: 1px solid white !important;
}
'''

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("""
        # CPS - 584 Deep Learning Project by Vignesh Yanamalamanda and Srija Tatineni
        
        This can be achieved  with the help of several open-source model: Stable Diffusiion XL Lightning | Parler TextToSpeec | WhisperSpeech | Hallo
        
        
        Thanks to Professor Mehdi For Inspiring Us to be creative while learning. and FYI. 4-5 seconds of audio will take ~5 minutes per inference, please be patient.
        """)
        with gr.Row(elem_id="column-names"):
            gr.Markdown("## 1. Load Image or Type")
            gr.Markdown("## 2. Load Voice or Type")
            gr.Markdown("## 3. Result")
        with gr.Group(elem_id="main-group"):
            with gr.Row():
                with gr.Column():
                    
                    portrait = gr.Image(
                        sources = ["upload"], 
                        type = "filepath",
                        format = "png",
                        elem_id = "image-block"
                    )

                    prompt_image = gr.Textbox(
                        label = "Generate image",
                        lines = 2,
                        max_lines = 2
                    )

                    gen_image_btn = gr.Button("Generate portrait (optional)")

                with gr.Column(elem_id="audio-column"):
                    
                    voice = gr.Audio(
                        type = "filepath",
                        elem_id = "audio-block"
                    )

                    preprocess_audio_file = gr.File(visible=False)
                    

                    with gr.Tab("Parler TTS", elem_id="parler-tab"):

                        prompt_audio = gr.Textbox(
                            label = "Text to synthetize",
                            lines = 3,
                            max_lines = 3,
                            elem_id = "text-synth"
                        )

                        voice_description = gr.Textbox(
                            label = "Voice description",
                            lines = 3,
                            max_lines = 3,
                            elem_id = "voice-desc"
                        )

                        gen_voice_btn = gr.Button("Generate voice (optional)")
                    
                    with gr.Tab("WhisperSpeech", elem_id="whisperspeech-tab"):
                        prompt_audio_whisperspeech = gr.Textbox(
                            label = "Text to synthetize",
                            lines = 2,
                            max_lines = 2,
                            elem_id = "text-synth-wsp"
                        )
                        audio_to_clone = gr.Audio(
                            label = "Voice to clone",
                            type = "filepath",
                            elem_id = "audio-clone-elm"
                        )
                        gen_wsp_voice_btn = gr.Button("Generate voice clone (optional)")
                
                with gr.Column(elem_id="result-column"): 
                    
                    result = gr.Video(
                        elem_id="video-block"
                    )
                    
                    submit_btn = gr.Button("Go talking Portrait !", elem_id="main-submit")
        
        with gr.Row(elem_id="pro-tips"):
            gr.Markdown("""
            # Project done in Summer 2024 at University of Dayton, Dayton, OH


            """)

            gr.Markdown("""
            # Application is made on Gradio and Follow up with files for reference

            """)

    portrait.upload(
        fn = convert_user_uploded_webp,
        inputs = [portrait],
        outputs = [portrait],
        queue = False,
        show_api = False
    )

    voice.upload(
        fn = check_mp3,
        inputs = [voice],
        outputs = [voice, preprocess_audio_file],
        queue = False,
        show_api = False
    )

    voice.clear(
        fn = clear_audio_elms,
        inputs = None,
        outputs = [preprocess_audio_file],
        queue = False,
        show_api = False
    )

    gen_image_btn.click(
        fn = generate_portrait,
        inputs = [prompt_image],
        outputs = [portrait],
        queue = False,
        show_api = False
    )

    gen_voice_btn.click(
        fn = generate_voice_with_parler,
        inputs = [prompt_audio, voice_description],
        outputs = [voice, preprocess_audio_file],
        queue = False,
        show_api = False
    )

    gen_wsp_voice_btn.click(
        fn = get_whisperspeech,
        inputs = [prompt_audio_whisperspeech, audio_to_clone],
        outputs = [voice, preprocess_audio_file],
        queue = False,
        show_api = False
    )

    submit_btn.click(
        fn = generate_talking_portrait,
        inputs = [portrait, voice],
        outputs = [result],
        show_api = False
    )
        

demo.queue(max_size=100).launch(show_error=True, show_api=False, share =True)