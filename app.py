import streamlit as st
import boto3
from PIL import Image
import io


aws_access_key_id = st.secrets["aws_access_key_id"]
aws_secret_access_key = st.secrets["aws_secret_access_key"]

rekognition_client = boto3.client('rekognition', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name='us-east-1')

#st.write("hello world!!!")

def compare_faces(image_image_bytes, target_image_bytes, threshold=80):
    try:
        response = rekognition_client.compare_faces(
            SourceImage={'Bytes': image_image_bytes},
            TargetImage={'Bytes': target_image_bytes},
            SimilarityThreshold=threshold
        )

        if len(response['FaceMatches']) > 0:
            return True, response['FaceMatches']
        else:
            return False, None
    except Exception as e:
        st.error(f"Erro ao comparar as imagens: {str(e)}")
        return False, None


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
uploaded_document = st.file_uploader("Faça o upload da foto de um documento ou da foto em um documento", type=["jpg", "jpeg", "png"])

if uploaded_document is not None:
    imagem_documento = Image.open(uploaded_document)
    st.image(imagem_documento, caption="Imagem recebida", use_container_width=True)
    imagem_documento_bytes = uploaded_document.getvalue()

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

            # Armazena a foto para que ela seja usada na comparação dos outros passos
            st.session_state.imagem_documento_bytes = imagem_documento_bytes
    else:
        st.warning("Nenhum rosto detectado na imagem.")


if 'imagem_documento_bytes' in st.session_state:
    st.write("Verificação de identidade")
    uploaded_selfie = st.file_uploader("Faça o upload de uma foto do seu rosto", type=["jpg", "jpeg", "png"])

    if uploaded_selfie is not None:
        imagem_selfie = Image.open(uploaded_selfie)
        st.image(imagem_selfie, caption="Foto de verificação carregada", use_container_width=True)
        imagem_selfie_bytes = uploaded_selfie.getvalue()

        # Threshold configurável
        confidence_threshold = st.slider(
            "Escolha o nível de confiança mínimo para considerar um match",
            min_value=0,
            max_value=100,
            value=80,  # Valor inicial de confiança (80%)
            step=1
        )
        match_found, face_matches = compare_faces(st.session_state.image1_bytes, imagem_selfie_bytes, threshold=confidence_threshold)
        if match_found:
            st.success(f"Identidade verificada com sucesso! Similaridade: {face_matches[0]['Similarity']:.2f}%")
        else:
            st.warning("Identidade não confirmada.")



