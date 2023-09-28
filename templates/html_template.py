css = '''
<style>
[data-testid="stSidebarUserContent"]{
    background-color: #AC9FAD;
 }
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://media.licdn.com/dms/image/D4D12AQF9Pp9b6GDNAA/article-cover_image-shrink_600_2000/0/1688726068486?e=2147483647&v=beta&t=OolhPqg0jqMA2t9FVEt1A01WAejqQgOztWlZ2bkzAwg" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://cdn.pixabay.com/photo/2017/01/31/21/23/avatar-2027366_1280.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''
