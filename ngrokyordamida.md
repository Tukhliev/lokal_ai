# Lokal AI Yordamchi (Matnli Chat + Rasm Tahlili)
# Muallif: ChatGPT (GPT-4o)
# OS: Kali Linux yoki Ubuntu Server
# CPU: Intel Iris Xe (GPU shart emas)
# GUI: Gradio
# Model: Mistral (matn) + BLIP (rasm)

# =============================
# 📦 1. O‘rnatish: Ubuntu Server uchun
# =============================
# Step-by-step to‘liq o‘rnatish uchun terminalda quyidagilarni bajaring:

# 1.1. Python, pip, curl, venv, git
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git curl unzip

# 1.2. Ollama (matnli AI model uchun)
curl -fsSL https://ollama.com/install.sh | sh
ollama run mistral

# 1.3. Loyiha papkasini yaratish
mkdir -p ~/lokal_ai_yordamchi && cd ~/lokal_ai_yordamchi

# 1.4. Virtual muhit yaratish va faollashtirish
python3 -m venv venv
source venv/bin/activate

# 1.5. Kutubxonalarni o‘rnatish
pip install torch transformers gradio pillow

# =============================
# 📂 2. Fayl: ai_yordamchi.py
# =============================
# Ushbu kodni 'ai_yordamchi.py' fayliga joylashtiring:

import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import subprocess

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def ai_chat(prompt):
    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )
    return result.stdout.decode().strip()

def image_caption(image):
    image = image.convert('RGB')
    inputs = processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def respond(message, chat_history):
    response = ai_chat(message)
    chat_history.append((message, response))
    return "", chat_history

with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("""
    # 🤖 Shaxsiy Lokal AI Yordamchi
    Offline ishlaydi · Matnli suhbat · Rasm tahlili
    """)

    with gr.Tab("💬 Matnli suhbat"):
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="Savolingizni yozing")
        send_btn = gr.Button("Yuborish")
        state = gr.State([])
        send_btn.click(fn=respond, inputs=[msg, state], outputs=[msg, chatbot])

    with gr.Tab("🖼 Rasm tahlili"):
        image_input = gr.Image(type="pil")
        caption_output = gr.Textbox(label="Rasm tavsifi")
        analyze_btn = gr.Button("Tahlil qilish")
        analyze_btn.click(fn=image_caption, inputs=image_input, outputs=caption_output)

    demo.queue().launch(server_name="127.0.0.1", server_port=7860, share=False)

# =============================
# 🌐 3. NGROK bilan HTTPS orqali kirish (bepul)
# =============================
# 3.1. Ngrok’ni yuklab oling va o‘rnating:
cd ~
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-stable-linux-amd64.zip
unzip ngrok-stable-linux-amd64.zip
sudo mv ngrok /usr/local/bin

# 3.2. Ngrok.com saytida ro‘yxatdan o‘ting va auth token oling:
# https://dashboard.ngrok.com/get-started/setup
ngrok config add-authtoken 2zthHWhIxeVLELWHQ13aGbzyv3L_29rUWqHEUChXX4BKV5jy7

# 3.3. Ngrok orqali 7860-portni HTTPS bilan oching:
ngrok http 7860

# 3.4. Natijada HTTPS link olasiz:
# Forwarding: https://abcd-1234.ngrok.io -> http://localhost:7860
# Bu link orqali boshqa foydalanuvchilar GUI’ga kira olishadi

# =============================
# 🚀 4. Har safar server yoqilganda AI va Ngrok avtomatik ishga tushishi
# =============================

# 4.1. AI uchun servis fayl:
# sudo nano /etc/systemd/system/ai-yordamchi.service

[Unit]
Description=Lokal AI Yordamchi
After=network.target

[Service]
User=ubuntu  # kerak bo‘lsa o‘zgartiring
WorkingDirectory=/home/ubuntu/lokal_ai_yordamchi
ExecStart=/home/ubuntu/lokal_ai_yordamchi/venv/bin/python3 ai_yordamchi.py
Restart=always

[Install]
WantedBy=multi-user.target

# 4.2. Ngrok uchun servis fayl:
# sudo nano /etc/systemd/system/ngrok-ai.service

[Unit]
Description=Ngrok tunnel for AI GUI
After=network.target

[Service]
ExecStart=/usr/local/bin/ngrok http 7860
Restart=always
User=ubuntu

[Install]
WantedBy=multi-user.target

# 4.3. Servislarni yoqing va ishga tushiring:
sudo systemctl daemon-reload
sudo systemctl enable ai-yordamchi.service ngrok-ai.service
sudo systemctl start ai-yordamchi.service ngrok-ai.service

# =============================
# ✅ Tayyor! Endi sizda: offline ishlaydigan, HTTPS orqali kiriladigan AI yordamchi mavjud.

