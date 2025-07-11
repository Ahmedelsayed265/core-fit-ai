FROM python:3.12-slim

# تثبيت مكتبات النظام المطلوبة لـ opencv
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# إعداد مجلد التطبيق
WORKDIR /app

# نسخ ملفات المشروع
COPY . /app

# تثبيت باقات Python
RUN pip install --upgrade pip && pip install -r requirements.txt

# تعيين البورت الافتراضي (يُستخدم من Railway تلقائيًا)
ENV PORT=8080

# تشغيل التطبيق
CMD ["python", "main.py"]
