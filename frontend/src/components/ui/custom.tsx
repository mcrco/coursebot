import React, { useState, useRef, useEffect } from "react";
import { Send, Loader2 } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar } from "@/components/ui/avatar";

// Define Message type to match your existing code
type Message = {
    id: string;
    role: string;
    content: string;
};

// ChatContainer component
const ChatContainer: React.FC<{
    className?: string;
    children: React.ReactNode;
}> = ({ className, children }) => {
    return (
        <div className={`flex flex-col h-full bg-background ${className}`}>
            {children}
        </div>
    );
};

// ChatMessages component
const ChatMessages: React.FC<{
    messages: Message[];
    children: React.ReactNode;
}> = ({ children }) => {
    return <div className="flex-1 overflow-hidden">{children}</div>;
};

// MessageList component
const MessageList: React.FC<{
    messages: Message[];
}> = ({ messages }) => {
    const messagesEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    return (
        <ScrollArea className="h-full p-4">
            <div className="space-y-4">
                {messages.map((message) => (
                    <div
                        key={message.id}
                        className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                    >
                        <div
                            className={`flex gap-3 max-w-xs md:max-w-md lg:max-w-lg ${message.role === "user" ? "flex-row-reverse" : "flex-row"
                                }`}
                        >
                            <Avatar className={`h-8 w-8 ${message.role === "assistant" ? "bg-primary" : "bg-secondary"}`}>
                                {message.role === "assistant" ? "A" : "U"}
                            </Avatar>
                            <div>
                                <Card className={`${message.role === "user" ? "bg-primary text-primary-foreground" : ""}`}>
                                    <CardContent className="p-3">
                                        <p>{message.content}</p>
                                    </CardContent>
                                </Card>
                            </div>
                        </div>
                    </div>
                ))}
                <div ref={messagesEndRef} />
            </div>
        </ScrollArea>
    );
};

// ChatForm component
const ChatForm: React.FC<{
    className?: string;
    isPending: boolean;
    handleSubmit: (e: React.FormEvent) => void;
    children: () => React.ReactNode;
}> = ({ className, isPending, handleSubmit, children }) => {
    return (
        <div className={`p-3 border-t ${className}`}>
            <form onSubmit={handleSubmit} className="flex gap-2">
                {children()}
            </form>
        </div>
    );
};

// MessageInput component
const MessageInput: React.FC<{
    value: string;
    onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
    stop?: () => void;
    isGenerating: boolean;
}> = ({ value, onChange, stop, isGenerating }) => {
    return (
        <>
            <Input
                placeholder="Type your message..."
                value={value}
                onChange={onChange}
                className="flex-1"
                disabled={isGenerating}
            />
            <Button type="submit" size="icon" disabled={isGenerating || !value.trim()}>
                {isGenerating ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            </Button>
            {isGenerating && stop && (
                <Button type="button" variant="outline" onClick={stop}>
                    Stop
                </Button>
            )}
        </>
    );
};

// PromptSuggestions component
const PromptSuggestions: React.FC<{
    label: string;
    suggestions: string[];
    append: (message: { role: string; content: string }) => void;
}> = ({ label, suggestions, append }) => {
    return (
        <div className="space-y-4">
            <p className="text-center text-muted-foreground">{label}</p>
            <div className="flex flex-col gap-2">
                {suggestions.map((suggestion, index) => (
                    <Button
                        key={index}
                        variant="outline"
                        className="text-left h-auto whitespace-normal p-4"
                        onClick={() => append({ role: "user", content: suggestion })}
                    >
                        {suggestion}
                    </Button>
                ))}
            </div>
        </div>
    );
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

    // Placeholder for stop function
    const stop = () => {
        // Implement stop functionality if needed
        setLoading(false);
    };

    return (
        <ChatContainer className="h-screen max-h-full flex flex-col gap-6 justify-center">
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

// Export all components to match the import structure of the original code
export { ChatContainer, ChatForm, ChatMessages, MessageList, MessageInput, PromptSuggestions };
