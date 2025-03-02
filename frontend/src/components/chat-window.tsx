import React, { useState } from "react";
import { ChatContainer, ChatForm, ChatMessages, MessageList, MessageInput, PromptSuggestions } from "@/components/ui/custom";

type Message = {
    id: string;
    role: string;
    content: string;
};

export const ChatWindow = () => {
    const [messages, setMessages] = useState<Array<Message>>([]);
    const [input, setInput] = useState("");
    const [isLoading, setLoading] = useState(false);

    const API_URL_BASE = import.meta.env.VITE_API_BASE_URL;
    const isEmpty = messages.length === 0;

    const getCompletion = async (queryMessage: Message) => {
        const payload = {
            messages: [
                ...messages.map((message) => ({
                    role: message.role,
                    content: message.content,
                })),
                { role: queryMessage.role, content: queryMessage.content },
            ],
        };

        const response = await fetch(API_URL_BASE + "/api/query", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            throw new Error("Network response was not ok");
        }

        const reader = response.body?.getReader();
        if (!reader) {
            throw new Error("No reader available");
        }

        const decoder = new TextDecoder();
        let content = "";
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            console.log(chunk);
            content += chunk;

            const responseMessage = {
                id: (messages.length + 1).toString(),
                role: "assistant",
                content: content,
            };

            setMessages([...messages, queryMessage, responseMessage]);
        }

        setLoading(false);
    };

    const appendMessage = (message: { role: string; content: string }) => {
        const newMessage = {
            id: messages.length.toString(),
            role: message.role,
            content: message.content,
        };
        setMessages([...messages, newMessage]);
        setLoading(true);
        getCompletion(newMessage);
    };

    const submitMessage = (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim()) return;
        appendMessage({ role: "user", content: input });
        setInput("");
    };

    const stop = () => {
        setLoading(false);
    };

    return (
        <ChatContainer className="h-screen max-h-full flex flex-col gap-6 justify-center w-full sm:w-1/2">
            {isEmpty ? (
                <h1 className="text-3xl font-bold text-center">Caltech Course Bot</h1>
            ) : (
                <h1 className="text-xl font-bold text-center">Caltech Course Bot</h1>
            )}

            {isEmpty ? (
                <div className="space-y-6">
                    <PromptSuggestions
                        label="Don't know what to ask? Try these prompts!"
                        append={appendMessage}
                        suggestions={[
                            "Are there any tennis courses at Caltech?",
                            "I like philosophizing time travel. Are there any classes about this?",
                            "What do students think about Caltech's intro CS courses?",
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

            <ChatForm className="mt-auto" isPending={isLoading} handleSubmit={submitMessage}>
                {() => (
                    <MessageInput
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        stop={stop}
                        isGenerating={isLoading}
                    />
                )}
            </ChatForm>
        </ChatContainer>
    );
};