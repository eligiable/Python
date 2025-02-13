from flask import render_template, redirect, url_for, request, flash, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from app import app, db, celery
from app.models import User, Post, Analytics
from app.forms import LoginForm, RegistrationForm, PostForm
from app.tasks import send_notification

@app.route('/')
@login_required
def dashboard():
    posts = Post.query.filter_by(user_id=current_user.id).all()
    return render_template('dashboard.html', posts=posts)

@app.route('/api/posts', methods=['GET', 'POST'])
def api_posts():
    if request.method == 'GET':
        posts = Post.query.all()
        return jsonify([{'id': post.id, 'content': post.content} for post in posts])
    elif request.method == 'POST':
        data = request.get_json()
        post = Post(content=data['content'], scheduled_time=data['scheduled_time'])
        db.session.add(post)
        db.session.commit()
        return jsonify({'message': 'Post created successfully'}), 201