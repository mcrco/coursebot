import { useState } from "react"
import { Chat } from "./ui/chat";
import { Message } from "./ui/chat-message";

export const ChatWindow = () => {
    const [messages, setMessages] = useState<Array<Message>>([]);
    const [input, setInput] = useState("");
    const [isLoading, setLoading] = useState(false);

    const getCompletion = (lastMessage: Message) => {
        const payload = {
            'messages': [...messages.map(message => ({
                role: message.role == 'user' ? 'user' : 'system',
                content: message.content
            })), { role: 'user', content: input }]
        };

        fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
        })
            .then((response) => response.json())
            .then((data) => {
                setMessages([...messages, lastMessage, data.response]);
                setLoading(false);
            })
    }

    const submitMessage = (e: { preventDefault?: (() => void) } | undefined) => {
        if (e !== undefined && e?.preventDefault !== undefined) {
            e.preventDefault();
        }
        const lastMessage: Message = { id: messages.length.toString(), role: 'user', content: input };
        setInput('');
        setMessages([...messages, lastMessage]);
        setLoading(true);
        getCompletion(lastMessage);
    }

    return (
        <Chat
            messages={messages}
            input={input}
            handleInputChange={(e) => setInput(e.target.value)}
            handleSubmit={submitMessage}
            isGenerating={isLoading}
        />
    )
}
