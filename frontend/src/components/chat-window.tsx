import { useState } from "react"
import { ChatContainer, ChatForm, ChatMessages } from "./ui/chat";
import { Message } from "./ui/chat-message";
import { PromptSuggestions } from "./ui/prompt-suggestions";
import { MessageList } from "./ui/message-list";
import { MessageInput } from "./ui/message-input";

export const ChatWindow = () => {
    const [messages, setMessages] = useState<Array<Message>>([]);
    const [input, setInput] = useState("");
    const [isLoading, setLoading] = useState(false);

    const isEmpty = messages.length === 0;

    const getCompletion = (queryMessage: Message) => {
        const payload = {
            'messages': [...messages.map(message => ({
                role: message.role,
                content: message.content
            })), { role: queryMessage.role, content: queryMessage.content }]
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
                setMessages([...messages, queryMessage, data.response]);
                setLoading(false);
            })
    }

    const appendMessage = (message: { role: string, content: string }) => {
        const newMessage = {
            id: messages.length.toString(),
            role: message.role,
            content: message.content
        };
        setMessages([...messages, newMessage])
        setLoading(true);
        getCompletion(newMessage);
    }

    const submitMessage = (e: { preventDefault?: (() => void) } | undefined) => {
        if (e !== undefined && e?.preventDefault !== undefined) {
            e.preventDefault();
        }
        appendMessage({ role: 'user', content: input });
        setInput('');
    }

    return (
        <ChatContainer className="h-screen max-h-full flex flex-col gap-6 justify-center">
            {
                isEmpty ? (
                    <h1 className='text-3xl font-bold text-center'>Caltech Course Bot</h1>
                ) :
                    (
                        <h1 className='text-xl font-bold text-center'>Caltech Course Bot</h1>
                    )

            }

            {isEmpty ? (
                <div className="space-y-6">
                    <PromptSuggestions
                        label="Don't know what to ask? Try these prompts!"
                        append={appendMessage}
                        suggestions={[
                            "Are there any tennis courses at Caltech?",
                            "I like philosophizing time travel. Are there any classes about this?",
                            "What do students think about Caltech's intro CS courses?"
                        ]}
                    />
                </div>
            ) : null}

            {!isEmpty ? (
                <div className="space-y-6 max-h-3/4 overflow-y-auto px-4">
                    <ChatMessages messages={messages}>
                        <MessageList messages={messages} />
                    </ChatMessages>
                </div>
            ) : null}

            <ChatForm
                className="mt-auto"
                isPending={isLoading}
                handleSubmit={submitMessage}
            >{() => (
                <MessageInput
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    stop={stop}
                    isGenerating={isLoading}
                />
            )}
            </ChatForm>
        </ChatContainer>
    )
}
