import streamlit as st
import boto3
from PIL import Image
import io


aws_access_key_id = st.secrets["aws_access_key_id"]
aws_secret_access_key = st.secrets["aws_secret_access_key"]

rekognition_client = boto3.client('rekognition', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name='us-east-1')

#st.write("hello world!!!")


def detect_document_face(image_bytes):
    try:
        # 'ALL' indica que é para retornar todos os atributos faciais
        faces_response = rekognition_client.detect_faces(Image={'Bytes': image_bytes}, Attributes=['ALL'])
        if len(faces_response['FaceDetails']) > 0:
            return True, faces_response['FaceDetails']
        else:
            return False, None
    except Exception as e:
        st.error(f"Erro ao processar a imagem: {str(e)}. Verifique o arquivo")
        return False, None
    

# IDENTIFICAÇÃO DA FOTO DE UM DOCUMENTO
st.title("Identificação de Rosto com Rekognition")
uploaded_file = st.file_uploader("Faça o upload da foto de um documento ou da foto em um documento", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    imagem_documento = Image.open(uploaded_file)
    st.image(imagem_documento, caption="Imagem recebida", use_container_width=True)
    imagem_documento_bytes = uploaded_file.getvalue()
    has_face, face_details = detect_document_face(imagem_documento_bytes)
    if has_face:
        st.success("Rosto encontrado na imagem")
        for i, face in enumerate(face_details):
            st.write(f"Rosto {i+1}:")
            st.write(f"Idade estimada: {face['AgeRange']['Low']} - {face['AgeRange']['High']} anos")
            st.write(f"Emoções detectadas: {[emotion['Type'] for emotion in face['Emotions']]}")
            emotions = sorted(face['Emotions'], key=lambda x: x['Confidence'], reverse=True)
            most_confident_emotion = emotions[0]
            st.write(f"Emoção predominante: {most_confident_emotion['Type']}, Confiança: {most_confident_emotion['Confidence']:.2f}%")
    else:
        st.warning("Nenhum rosto detectado na imagem.")



