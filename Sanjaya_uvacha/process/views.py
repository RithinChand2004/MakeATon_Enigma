from django.shortcuts import render
from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from django.views.decorators.csrf import csrf_exempt
import azure.cognitiveservices.speech as speechsdk
from .models import Image
from .serializers import ImageSerializer
from django.http import JsonResponse, HttpResponseBadRequest
from azure.storage.blob import BlobServiceClient
import json
import base64
import os
import pyttsx3
import threading
import azure.ai.vision as sdk
import cv2
import numpy as np
import requests
from io import BytesIO
import mediapipe as mp
import azure.ai.vision as sdk
# Create your views here.


def text_to_speech(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.say(text)
    engine.runAndWait()

def speakOut(text):
    speech_thread = threading.Thread(target=text_to_speech, args=(text,))
    speech_thread.start()

def speak(request):
    text = (request.GET.get('text'))
    speakOut(text)
    return JsonResponse({'text': text})

@csrf_exempt
def uploadFrame(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_data_url = data['image']

            # Connect to your Azure Blob Storage account
            connection_string = "DefaultEndpointsProtocol=https;AccountName=bumble;AccountKey=c/mSmXiLPb7OK3KKC8WdnUPP81pCjg3ApavOfPHNvk+iPtVFfWd660mupLiJ3QIt/1sege3FYZuv+AStI/KEmA==;EndpointSuffix=core.windows.net"
            blob_service_client = BlobServiceClient.from_connection_string(
                connection_string)

            container_name = "data"
            container_client = blob_service_client.get_container_client(
                container_name)

            # Generate a unique blob name (e.g., based on timestamp or UUID)
            blob_name = "frame123.jpg"  # Replace with your naming logic

            # Decode the base64 image data and upload it to Azure Blob Storage
            image_bytes = base64.b64decode(image_data_url.split(',')[1])
            container_client.upload_blob(
                name=blob_name, data=image_bytes, overwrite=True)

            return JsonResponse({'message': 'Frame uploaded successfully'})
        except Exception as e:
            return HttpResponseBadRequest('Error uploading frame: ' + str(e))
    else:
        return HttpResponseBadRequest('Invalid request method')

def recognize_from_microphone():
    text_to_speech("How can I help you today?")
    output = ''
    # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
    speech_config = speechsdk.SpeechConfig(
        "a7bb5bb2551b4837a6183c173ad34e54", 'eastus')
    speech_config.speech_recognition_language = "en-US"

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config)

    print("Speak into your microphone.")
    speech_recognition_result = speech_recognizer.recognize_once_async().get()

    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(speech_recognition_result.text))
        output = speech_recognition_result.text
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(
            speech_recognition_result.no_match_details))
        output = "No speech could be recognized: {}".format(
            speech_recognition_result.no_match_details)
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        print("Speech Recognition canceled: {}".format(
            cancellation_details.reason))
        output = "Speech Recognition canceled: {}".format(
            cancellation_details.reason)
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(
                cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")
    return output


def home(request):
    context = {'textOutput': 0}
    return render(request, 'home.html')

def listenAndRecognize(request):
    text = recognize_from_microphone()
    return JsonResponse({'text': text})


@csrf_exempt
def uploadImg(request):
    # csrf tockens

    if request.method == 'POST':
        print(request.POST)
        print(request.FILES)
        title = request.POST['title']
        image = request.FILES['image']
        print(title)
        print(image)
        Image.objects.create(title=title, image=image)
    return render(request, 'home.html')


class ImageUploadView(APIView):
    parser_classes = (FileUploadParser,)

    def post(self, request, *args, **kwargs):
        print(request.data)
        file_serializer = ImageSerializer(data=request.data)
        print(file_serializer)
        if file_serializer.is_valid():
            file_serializer.save()
            return Response(file_serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

def findObjects(request):
    service_options = sdk.VisionServiceOptions("https://dristi.cognitiveservices.azure.com/",
                                               "ec8f8825967f482699df9b8080d3d826")

    vision_source = sdk.VisionSource(
        url="https://bumble.blob.core.windows.net/data/frame123.jpg")

    analysis_options = sdk.ImageAnalysisOptions()

    analysis_options.features = (
        # sdk.ImageAnalysisFeature.CAPTION |
        # sdk.ImageAnalysisFeature.DENSE_CAPTIONS |
        sdk.ImageAnalysisFeature.OBJECTS 
        # sdk.ImageAnalysisFeature.PEOPLE |
        # sdk.ImageAnalysisFeature.TEXT |
        # sdk.ImageAnalysisFeature.TAGS
    )

    analysis_options.language = "en"

    analysis_options.gender_neutral_caption = True

    image_analyzer = sdk.ImageAnalyzer(
        service_options, vision_source, analysis_options)

    result = image_analyzer.analyze()

    if result.reason == sdk.ImageAnalysisResultReason.ANALYZED:

        print(" Image height: {}".format(result.image_height))
        print(" Image width: {}".format(result.image_width))
        print(" Model version: {}".format(result.model_version))

        obj = ''
        if result.objects is not None:
            print(" Objects:")
            for object in result.objects:
                print("   '{}', {}, Confidence: {:.4f}".format(
                    object.name, object.bounding_box, object.confidence))
                if object.confidence > 0.5:
                    obj += object.name + ' '
        else:
            print("No objects detected.")
        
        speakOut(obj)
        return JsonResponse({'objects': obj})

        

    else:

        error_details = sdk.ImageAnalysisErrorDetails.from_result(result)
        print(" Analysis failed.")
        # print("   Error reason: {}".format(error_details.reason))
        # print("   Error code: {}".format(error_details.error_code))
        # print("   Error message: {}".format(error_details.message))