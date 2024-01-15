from sentence_transformers import SentenceTransformer # 이 라이브러리는 문장 또는 단락과 같은 텍스트 데이터를 수치화된 벡터로 변환하는 데 사용됩니다.
from qdrant_client import QdrantClient  # 이 라이브러리는 Qdrant 벡터 검색 엔진과의 상호 작용을 위해 사용됩니다. 
from qdrant_client.http.models import PointStruct, VectorParams, Distance # Qdrant 벡터 검색 엔진에서 데이터를 효율적으로 삽입, 검색 및 관리하는 데 필요한 도구입니다.

# Qdrant 클라이언트 초기화 (수정 필요)
client = QdrantClient(host="localhost", port=6333) 
collection_name = "text_collection"

# 텍스트를 벡터로 변환하기 위한 모델 초기화 (이 모델은 여러 언어에 대한 이해를 기반으로 하며, 문장이나 단락의 의미를 벡터 형태로 인코딩합니다.)
model = SentenceTransformer('all-MiniLM-L6-v2')

# 텍스트 파일 읽기 및 벡터 변환 (전체 텍스트 -> 하나의 벡터)
def read_and_encode(file_name):
    with open(file_name, 'r') as file:
        text = file.read()  # 전체 파일 내용을 하나의 문자열로 읽음
    vector = model.encode(text, convert_to_tensor=True)
    return text, vector

text1, vector1 = read_and_encode('test1.txt')
text2, vector2 = read_and_encode('test2.txt')

# Qdrant에 업로드 (test1.txt 및 test2.txt)
points = [
    PointStruct(id=0, vector=vector1.tolist(), payload={"text": text1, "source": "test1"}),
    PointStruct(id=1, vector=vector2.tolist(), payload={"text": text2, "source": "test2"})
] 

client.upsert(collection_name=collection_name, points=points)
print("Data uploaded to Qdrant.")

# Qdrant update
updated_text1, updated_vector1 = read_and_encode('updated_test1.txt')
updated_point1 = PointStruct(id=0, vector=updated_vector1.tolist(), payload={"text": updated_text1, "source": "test1"})
client.upsert(collection_name=collection_name, points=[updated_point1])
print("test1.txt data updated in Qdrant.")

# test1,2.txt의 내용과 메타데이터 출력
print("Contents and Metadata of test1.txt:")
print(f"ID: 1, Text: {text1}, Source: test1")
print(f"ID: 1, Vector: {vector1}, Source: test1")

print("Contents and Metadata of test2.txt:")
print(f"ID: 2, Text: {text2}, Source: test2")
print(f"ID: 2, Vector: {vector2}, Source: test2")





