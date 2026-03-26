import re

# Read the current index.html
with open('index.html', 'r', encoding='utf-8') as f:
    content = f.read()

# First, let's add CSS styles before </style>
css_addition = '''        .edit-btn {
            background: rgba(255, 255, 255, 0.1);
            border: none;
            border-radius: 6px;
            color: #a0a0a0;
            cursor: pointer;
            font-size: 14px;
            padding: 4px 8px;
            opacity: 0;
            transition: all 0.2s ease;
            margin-left: 8px;
        }

        .message:hover .edit-btn {
            opacity: 1;
        }

        .edit-btn:hover {
            background: rgba(255, 255, 255, 0.2);
            color: #fff;
        }

        .edit-textarea {
            width: 100%;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            color: #fff;
            font-family: inherit;
            font-size: 0.95rem;
            padding: 8px 12px;
            resize: vertical;
            min-height: 60px;
        }

        .edit-buttons {
            display: flex;
            gap: 8px;
            margin-top: 8px;
        }

        .edit-save-btn, .edit-cancel-btn {
            padding: 6px 12px;
            border-radius: 6px;
            border: none;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s ease;
        }

        .edit-save-btn {
            background: #a8edea;
            color: #1a1a2e;
        }

        .edit-save-btn:hover {
            background: #8dd9d6;
        }

        .edit-cancel-btn {
            background: rgba(255, 255, 255, 0.1);
            color: #a0a0a0;
        }

        .edit-cancel-btn:hover {
            background: rgba(255, 255, 255, 0.2);
            color: #fff;
        }

        .edit-marker {
            font-size: 11px;
            color: #a0a0a0;
            margin-left: 8px;
        }
'''

# Insert CSS before </style>
content = content.replace('    </style>', css_addition + '    </style>')

# Now add the edit functionality after appendTag function
edit_function = '''
            // ---- メッセージ編集機能 ----
            function editMessage(msgDiv, originalText) {
                const bubble = msgDiv.querySelector('.bubble');
                
                const textarea = document.createElement('textarea');
                textarea.className = 'edit-textarea';
                textarea.value = originalText;
                textarea.rows = 3;
                
                const btnContainer = document.createElement('div');
                btnContainer.className = 'edit-buttons';
                
                const saveBtn = document.createElement('button');
                saveBtn.textContent = '保存';
                saveBtn.className = 'edit-save-btn';
                
                const cancelBtn = document.createElement('button');
                cancelBtn.textContent = 'キャンセル';
                cancelBtn.className = 'edit-cancel-btn';
                
                btnContainer.appendChild(saveBtn);
                btnContainer.appendChild(cancelBtn);
                
                const originalContent = bubble.innerHTML;
                bubble.innerHTML = '';
                bubble.appendChild(textarea);
                bubble.appendChild(btnContainer);
                textarea.focus();
                
                saveBtn.addEventListener('click', () => {
                    const newText = textarea.value.trim();
                    if (newText && newText !== originalText) {
                        bubble.innerHTML = newText.replace(/\\n/g, '<br>');
                        const editMarker = document.createElement('span');
                        editMarker.className = 'edit-marker';
                        editMarker.textContent = '（編集済）';
                        bubble.appendChild(editMarker);
                    } else {
                        bubble.innerHTML = originalContent;
                    }
                });
                
                cancelBtn.addEventListener('click', () => {
                    bubble.innerHTML = originalContent;
                });
            }
'''

# Insert edit function after appendTag function
content = content.replace(
    '''            function appendTag(parent, cls, icon, text) {
                const span = document.createElement('span');
                span.className = `tag ${cls}`;
                span.textContent = `${icon} ${text}`;
                parent.appendChild(span);
            }''',
    '''            function appendTag(parent, cls, icon, text) {
                const span = document.createElement('span');
                span.className = `tag ${cls}`;
                span.textContent = `${icon} ${text}`;
                parent.appendChild(span);
            }''' + edit_function
)

# Now modify addMessage function to include edit button for user messages
old_addMessage = '''            // ---- メッセージをDOMに追加 ----
            function addMessage(role, text, thought, emotion, action) {
                const msg = document.createElement('div');
                msg.className = `message ${role}`;

                const avatarText = role === 'ai' ? 'A' : 'U';
                const avatar = document.createElement('div');
                avatar.className = `avatar ${role}`;
                avatar.textContent = avatarText;

                const bubble = document.createElement('div');
                bubble.className = 'bubble';
                bubble.textContent = text;
                // 改行だけ <br> に置換（テキストは既にエスケープ済み）
                bubble.innerHTML = bubble.innerHTML.replace(/\\n/g, '<br>');

                msg.appendChild(avatar);
                msg.appendChild(bubble);
                chatArea.appendChild(msg);

                // 内面状態タグ（AIのみ）
                if (role === 'ai' && (thought || emotion || action)) {
                    const tags = document.createElement('div');
                    tags.className = 'inner-state';
                    if (thought) appendTag(tags, 'thought', '💭', thought);
                    if (emotion) appendTag(tags, 'emotion', '✦', emotion);
                    if (action) appendTag(tags, 'action', '→', action);
                    chatArea.appendChild(tags);
                }

                scrollToBottom();
            }'''

new_addMessage = '''            // ---- メッセージをDOMに追加 ----
            function addMessage(role, text, thought, emotion, action) {
                const msg = document.createElement('div');
                msg.className = `message ${role}`;

                const avatarText = role === 'ai' ? 'A' : 'U';
                const avatar = document.createElement('div');
                avatar.className = `avatar ${role}`;
                avatar.textContent = avatarText;

                const bubble = document.createElement('div');
                bubble.className = 'bubble';
                bubble.textContent = text;
                bubble.innerHTML = bubble.innerHTML.replace(/\\n/g, '<br>');

                msg.appendChild(avatar);
                msg.appendChild(bubble);

                // 編集ボタン（ユーザーメッセージのみ）
                if (role === 'user') {
                    const editBtn = document.createElement('button');
                    editBtn.className = 'edit-btn';
                    editBtn.innerHTML = '✎';
                    editBtn.title = '編集';
                    editBtn.addEventListener('click', () => editMessage(msg, text));
                    msg.appendChild(editBtn);
                }

                chatArea.appendChild(msg);

                // 内面状態タグ（AIのみ）
                if (role === 'ai' && (thought || emotion || action)) {
                    const tags = document.createElement('div');
                    tags.className = 'inner-state';
                    if (thought) appendTag(tags, 'thought', '💭', thought);
                    if (emotion) appendTag(tags, 'emotion', '✦', emotion);
                    if (action) appendTag(tags, 'action', '→', action);
                    chatArea.appendChild(tags);
                }

                scrollToBottom();
            }'''

content = content.replace(old_addMessage, new_addMessage)

# Write the modified content back
with open('index.html', 'w', encoding='utf-8') as f:
    f.write(content)

print('Message edit feature added successfully!')
