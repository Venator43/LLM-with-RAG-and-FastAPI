# LLM-with-RAG-and-FastAPI
A simple API to query an LLM with RAG

# Deployment
sebelum anda menjalankan program saya, pertama anda harus memasukan api key hunggingface anda kedalam api.py seperti pada gambar berikut 

<img width="396" height="34" alt="image" src="https://github.com/user-attachments/assets/22ae7ffc-89fa-45cc-954b-6af8d30e70d0" />

Saya menyediakan 2 cara untuk menjalnkan program yang saya buat, satu melalui docker, dan satu tanpa menggunakan docker
untuk menjalankan program yang telah saya buat menggunakan docker, anda dapat dengan mudah menggunakan command "docker build -t nama_build ." untuk mem-build program didalam docker, kemudian, anda dapat menjalankan aplikasi dengan menggunakan command "docker run -p 8000:8002 nama_build", untuk mengakses API anda dapat memasuki URL "http://localhost:8002/docs#/"

Jika anda tidak menggunakan docker, pertama anda harus menginstall package torch dengan menggunakan command pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
dan package yang dibutuhkan dengan menggunakan command "pip install -r requirements_docker.txt" kemudian anda dapat menjalankan program API dengan cara menggunakan command "python api.py", sama seperti pada docker anda dapat mengakses API dengan cara memasuki URL : "http://localhost:8002/docs#/" 

# Contoh Program Berjalan Di API
Pada program yang saya buat, anda dapat mengubah context dr agent RAG dengan cara mengakses hitpoint POST: /api/context seperti gambar berikut (Note: Hanya bisa mengupload PDF)

<img width="1433" height="886" alt="image" src="https://github.com/user-attachments/assets/d3bceb06-c090-4a92-8cbb-7d5489630675" />

Untuk mengirim query ke LLM, anda dapat menggunakan hitpoint GET: /api/chat, LLM akan menjawab query anda sesuai dengan context RAG yang telah diberikan 

<img width="1333" height="911" alt="image" src="https://github.com/user-attachments/assets/d6ed7ae4-21ac-4a48-bc1f-1cd2f8d68bb1" />

<img width="971" height="481" alt="image" src="https://github.com/user-attachments/assets/f2a3b2eb-5f52-4a5d-9a79-c0fd8b6d011c" />
