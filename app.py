import streamlit as st
import boto3
from PIL import Image, ImageDraw
import io

# Criação do cliente Rekognition uma vez
aws_access_key_id = st.secrets["aws_access_key_id"]
aws_secret_access_key = st.secrets["aws_secret_access_key"]

rekognition_client = boto3.client('rekognition', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name='us-east-1')

def compare_faces(image_image_bytes, target_image_bytes, threshold=80):
    try:
        if not isinstance(image_image_bytes, bytes) or not isinstance(target_image_bytes, bytes):
            raise ValueError("Ambos os parâmetros devem ser do tipo 'bytes'.")

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


def detect_faces_in_crowd(image_bytes):
    try:
        response = rekognition_client.detect_faces(
            Image={'Bytes': image_bytes},
            Attributes=['ALL']  # Para retornar todos os atributos faciais
        )
        return response['FaceDetails']
    except Exception as e:
        st.error(f"Erro ao detectar rostos na multidão: {str(e)}")
        return []


def detect_document_face(image_bytes):
    try:
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
st.subheader("Upload do documento")
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

# VERIFICAÇÃO DE IDENTIDADE COM SELFIE DA CÂMERA
import streamlit as st
import boto3
from PIL import Image
import io

# Criação do cliente Rekognition
aws_access_key_id = st.secrets["aws_access_key_id"]
aws_secret_access_key = st.secrets["aws_secret_access_key"]

rekognition_client = boto3.client('rekognition', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name='us-east-1')

def compare_faces(image_bytes1, image_bytes2, threshold=80):
    try:
        if not isinstance(image_bytes1, bytes) or not isinstance(image_bytes2, bytes):
            raise ValueError("Ambos os parâmetros devem ser do tipo 'bytes'.")

        response = rekognition_client.compare_faces(
            SourceImage={'Bytes': image_bytes1},
            TargetImage={'Bytes': image_bytes2},
            SimilarityThreshold=threshold
        )

        if len(response['FaceMatches']) > 0:
            return True, response['FaceMatches']
        else:
            return False, None
    except Exception as e:
        st.error(f"Erro ao comparar as imagens: {str(e)}")
        return False, None


# VERIFICAÇÃO DE IDENTIDADE COM SELFIE DA CÂMERA
if 'imagem_documento_bytes' in st.session_state:
    st.subheader("Verificação de identidade - Tire uma selfie!")

    # Checkbox para ativar a câmera
    enable_camera = st.checkbox("Habilitar câmera")

    # Quando a câmera estiver habilitada, o usuário pode tirar a selfie
    selfie = None
    if enable_camera:
        selfie = st.camera_input("Tire uma foto")

    # Verificar se a selfie foi tirada
    if selfie is not None:
        st.write("A selfie foi tirada e armazenada!")

        # Armazenar os bytes da selfie no session_state para persistência
        st.session_state.imagem_selfie_camera = selfie.getvalue()

        # Exibir a selfie capturada
        imagem_selfie_camera = Image.open(io.BytesIO(st.session_state.imagem_selfie_camera))
        st.image(imagem_selfie_camera, caption="Selfie Capturada", use_container_width=True)

        # Obter os bytes da selfie para a comparação
        imagem_selfie_camera_bytes = st.session_state.imagem_selfie_camera

        # Threshold configurável
        confidence_threshold = st.slider(
            "Escolha o nível de confiança mínimo para considerar um match",
            min_value=0,
            max_value=100,
            value=80,  # Valor inicial de confiança (80%)
            step=1,
            key="confidence-threshold-selfie-from-camera"
        )

        # Comparar a selfie com a imagem de documento
        match_found, face_matches = compare_faces(st.session_state.imagem_documento_bytes, imagem_selfie_camera_bytes, threshold=confidence_threshold)

        if match_found:
            st.success(f"Identidade verificada com sucesso! Similaridade: {face_matches[0]['Similarity']:.2f}%")
        else:
            st.error("Identidade não confirmada.")
    else:
        st.write("Nenhuma selfie foi tirada ainda.")



# VERIFICAÇÃO DE IDENTIDADE COM SELFIE DE ARQUIVO
if 'imagem_documento_bytes' in st.session_state:
    st.subheader("Verificação de identidade - Faça o upload de uma selfie")
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
            step=1,
            key="confidence-threshold-selfie-from-file"
        )
        match_found, face_matches = compare_faces(st.session_state.imagem_documento_bytes, imagem_selfie_bytes, threshold=confidence_threshold)
        if match_found:
            st.success(f"Identidade verificada com sucesso! Similaridade: {face_matches[0]['Similarity']:.2f}%")
        else:
            st.error("Identidade não confirmada.")

# ENCONTRANDO A PESSOA ENTRE VÁRIAS OUTRAS
if 'imagem_documento_bytes' in st.session_state:
    st.subheader("Encontrar na multidão")
    uploaded_crowd = st.file_uploader("Faça o upload de uma foto com várias pessoas", type=["jpg", "jpeg", "png"])
    if uploaded_crowd is not None:
        imagem_crowd = Image.open(uploaded_crowd)
        st.image(imagem_crowd, caption="Foto da multidão carregada", use_container_width=True)
        imagem_crowd_bytes = uploaded_crowd.getvalue()

        # Detectar rostos na imagem da multidão
        faces_in_crowd = detect_faces_in_crowd(imagem_crowd_bytes)
        draw = ImageDraw.Draw(imagem_crowd)

        match_found = False
        for i, face_detail in enumerate(faces_in_crowd):
            bounding_box = face_detail['BoundingBox']

            # Calcular as coordenadas de recorte
            width, height = imagem_crowd.size
            left = int(bounding_box['Left'] * width)
            top = int(bounding_box['Top'] * height)
            right = int((bounding_box['Left'] + bounding_box['Width']) * width)
            bottom = int((bounding_box['Top'] + bounding_box['Height']) * height)

            # Recortar a face da imagem
            face_image = imagem_crowd.crop((left, top, right, bottom))

            # Converter a face recortada para bytes
            with io.BytesIO() as byte_io:
                face_image.save(byte_io, format='JPEG')
                face_bytes = byte_io.getvalue()

            st.write(f"Analisando face {i+1}")
            st.image(face_image, caption=f"Face {i+1}", width=50)

            response_crowd = compare_faces(st.session_state.imagem_documento_bytes, face_bytes, threshold=80)

            # Desenhar a caixa de rosto e indicar se há match ou não
            if response_crowd[0]:
                # Se houver match
                draw.rectangle([left, top, right, bottom], outline="green", width=5)
                match_found = True
                similarity = response_crowd[1][0]['Similarity']
                st.success(f"Match! Confiança para essa face: {response_crowd[1][0]['Similarity']:.2f}%")


            else:
                # Se não houver match
                draw.rectangle([left, top, right, bottom], outline="red", width=5)


        if match_found:
            st.success("Encontramos um match na multidão!")
        else:
            st.error("Nenhuma correspondência encontrada na multidão.")

        # Exibir a imagem com os retângulos
        st.image(imagem_crowd, caption="Foto da Multidão - Resultados da Comparação", use_container_width=True)
