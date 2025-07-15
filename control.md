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
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git curl

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

# --------- 1. Rasmni tahlil qiluvchi modelni yuklash ---------
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# --------- 2. Matnli AI chat (Ollama orqali Mistral) ---------
def ai_chat(prompt):
    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )
    return result.stdout.decode().strip()

# --------- 3. Rasm tahlili funksiyasi ---------
def image_caption(image):
    image = image.convert('RGB')
    inputs = processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# --------- 4. Matnli chat funksiyasi ---------
def respond(message, chat_history):
    response = ai_chat(message)
    chat_history.append((message, response))
    return "", chat_history

# --------- 5. Gradio GUI ---------
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

# --------- 6. Gradio’ni tashqi IP orqali ko‘rinadigan qilish ---------
demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

# =============================
# 🌐 3. Serverga brauzer orqali kirish
# =============================
# Kompyuteringizda brauzer oching va quyidagiga kiring:
# http://server-ip:7860

# Agar kerak bo‘lsa, 7860-portni oching:
# sudo ufw allow 7860

# =============================
# 🚀 4. Har safar server yoqilganda avtomatik ishga tushirish (systemd)
# =============================
# 4.1. Servis fayl yarating:
# sudo nano /etc/systemd/system/ai-yordamchi.service

# 4.2. Quyidagini joylashtiring:

# -------------------------------------------
[Unit]
Description=Lokal AI Yordamchi
After=network.target

[Service]
User=ubuntu  # <== kerak bo‘lsa foydalanuvchi nomini o‘zgartiring
WorkingDirectory=/home/ubuntu/lokal_ai_yordamchi
ExecStart=/home/ubuntu/lokal_ai_yordamchi/venv/bin/python3 ai_yordamchi.py
Restart=always

[Install]
WantedBy=multi-user.target
# -------------------------------------------

# 4.3. Servisni faollashtiring va ishga tushiring:
sudo systemctl daemon-reexec
sudo systemctl daemon-reload
sudo systemctl enable ai-yordamchi.service
sudo systemctl start ai-yordamchi.service

# 4.4. Holatini tekshiring:
sudo systemctl status ai-yordamchi.service

# ✅ Endi har safar server yoqilganda avtomatik ishga tushadi.

# =============================
# ✅ Tayyor! Endi sizda: offline, cheksiz rasm + chat bilan ishlaydigan AI yordamchi bor.

