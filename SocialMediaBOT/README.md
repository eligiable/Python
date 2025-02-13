# **Social Media Auto BOT**

A **Flask-based web application** to automate social media postings across multiple platforms (Twitter, Facebook, Instagram, LinkedIn, Pinterest, YouTube, etc.). The bot includes a **web-based dashboard** for managing posts, **user authentication**, **analytics tracking**, and **scheduling capabilities**. It is designed to be **scalable**, **performant**, and **easy to deploy** using Docker.

---

## **Features**

1. **Multi-Platform Support**:
   - Post to **Twitter**, **Facebook**, **Instagram**, **LinkedIn**, **Pinterest**, and **YouTube**.
   - Handle **text**, **images**, and **videos**.

2. **Web-Based Dashboard**:
   - Schedule posts.
   - View and manage scheduled posts.
   - Track post performance (likes, shares, clicks).

3. **User Authentication**:
   - User registration and login.
   - Role-based access control (admin and user roles).

4. **Analytics**:
   - Real-time analytics using WebSocket or AJAX.
   - Track post performance across platforms.

5. **Email Notifications**:
   - Notify users when posts are published or fail.

6. **API for External Integration**:
   - Expose RESTful APIs for scheduling posts and fetching analytics.

7. **Docker Deployment**:
   - Containerized for easy deployment using Docker and Docker Compose.

---

## **Technologies Used**

- **Backend**: Flask, Celery, Redis, SQLAlchemy
- **Frontend**: HTML, CSS (Bootstrap), Jinja2
- **Database**: PostgreSQL (or SQLite for development)
- **APIs**: Twitter API, Facebook API, Instagram API, LinkedIn API, Pinterest API, YouTube API
- **Deployment**: Docker, Docker Compose, Gunicorn

---

## **Installation**

### **Prerequisites**

1. **Python 3.9+**
2. **Docker** and **Docker Compose**
3. **API Keys** for all supported platforms (Twitter, Facebook, Instagram, etc.)

### **Steps**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/socialmedia-bot.git
   cd socialmedia-bot
   ```

2. **Set Up Environment Variables**:
   Create a `.env` file in the root directory and add the following variables:
   ```plaintext
   SECRET_KEY=your_secret_key
   DATABASE_URL=postgresql://user:password@db:5432/socialmedia
   CELERY_BROKER_URL=redis://redis:6379/0
   CELERY_RESULT_BACKEND=redis://redis:6379/0
   MAIL_SERVER=smtp.gmail.com
   MAIL_PORT=587
   MAIL_USE_TLS=True
   MAIL_USERNAME=your_email@gmail.com
   MAIL_PASSWORD=your_email_password
   ```
   If you have Python installed, you can generate a SECRET_KEY directly in your terminal:
   ```
   python -c "import secrets; print(secrets.token_hex(32))"
   ```

3. **Build and Run with Docker**:
   ```bash
   docker-compose up --build
   ```

4. **Access the Application**:
   Open your browser and go to `http://localhost:5000`.

---

## **Usage**

### **1. User Registration and Login**
- Register a new account or log in using existing credentials.
- Admins can access additional features like user management.

### **2. Dashboard**
- View all scheduled posts.
- Create new posts with text, images, or videos.
- Schedule posts for specific times.

### **3. Analytics**
- Track post performance (likes, shares, clicks) in real-time.

### **4. Email Notifications**
- Receive email notifications when posts are published or fail.

### **5. API Integration**
- Use the RESTful API to schedule posts or fetch analytics programmatically.

---

## **API Endpoints**

### **Posts**
- **GET** `/api/posts` - Fetch all posts.
- **POST** `/api/posts` - Create a new post.

### **Analytics**
- **GET** `/api/analytics` - Fetch analytics data for all posts.

---

## **Project Structure**

```
socialmedia-bot/
│
├── app/
│   ├── __init__.py
│   ├── models.py
│   ├── routes.py
│   ├── tasks.py
│   ├── templates/
│   │   ├── base.html
│   │   ├── dashboard.html
│   │   ├── login.html
│   │   ├── register.html
│   │   └── analytics.html
│   └── static/
│       └── styles.css
│
├── config.py
├── run.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```