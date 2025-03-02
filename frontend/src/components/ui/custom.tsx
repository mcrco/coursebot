import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Send, Loader2 } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";

type Message = {
    id: string;
    role: string;
    content: string;
};

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

const ChatMessages: React.FC<{
    messages: Message[];
    children: React.ReactNode;
}> = ({ children }) => {
    return <div className="flex-1 overflow-hidden">{children}</div>;
};

const MessageList: React.FC<{
    messages: Message[];
}> = ({ messages }) => {
    const messagesEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    return (
        <ScrollArea className="h-full p-4">
            <div className="space-y-6 mx-auto text-sm sm:text-sm md:text-base lg:text-md">
                {messages.map((message) => (
                    <div
                        key={message.id}
                        className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                    >
                        <div
                            className={`flex gap-3 w-full ${message.role === "user" ? "w-auto flex-row-reverse" : "flex-row"
                                }`}
                        >
                            <Card className={`${message.role === "user" ? "w-auto rounded-lg border" : "border-none w-full"} `}>
                                <CardContent className={`markdown ${message.role === "user" ? "p-4 pb-0 w-auto" : "p-0"}`}>
                                    <ReactMarkdown
                                        remarkPlugins={[remarkGfm]}
                                        className="prose dark:prose-invert max-w-none"
                                    >
                                        {message.content}
                                    </ReactMarkdown>
                                </CardContent>
                            </Card>
                        </div>
                    </div>
                ))}
                <div ref={messagesEndRef} />
            </div>
        </ScrollArea>
    );
};

const ChatForm: React.FC<{
    className?: string;
    isPending: boolean;
    handleSubmit: (e: React.FormEvent) => void;
    children: () => React.ReactNode;
}> = ({ className, handleSubmit, children }) => {
    return (
        <div className={`p-4 ${className}`}>
            <div className="p-2 rounded border">
                <form onSubmit={handleSubmit} className="flex gap-2">
                    {children()}
                </form>
            </div>
        </div>
    );
};

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
                className="flex-1 border-none focus-visible:ring-0"
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

const PromptSuggestions: React.FC<{
    label: string;
    suggestions: string[];
    append: (message: { role: string; content: string }) => void;
}> = ({ label, suggestions, append }) => {

    const [isMobile, setMobile] = useState(window.innerWidth < 768);

    useEffect(() => {
        const handleResize = () => {
            setMobile(window.innerWidth < 768);
        };

        window.addEventListener("resize", handleResize);
        return () => window.removeEventListener("resize", handleResize);
    }, []);

    if (isMobile) {
        return (
            <div className="space-y-4 p-6">
                <p className="text-center text-muted-foreground">{label}</p>
                <div className="flex flex-col justify-center gap-4">
                    {suggestions.map((suggestion, index) => (
                        <Button
                            key={index}
                            variant="outline"
                            className="text-center h-auto p-4 flex items-center whitespace-normal"
                            onClick={() => append({ role: "user", content: suggestion })}
                        >
                            {suggestion}
                        </Button>
                    ))}
                </div>
            </div>
        );
    }

    return (
        <div className="space-y-4 p-6">
            <p className="text-center text-muted-foreground">{label}</p>
            <div className="flex flex-row justify-center gap-4">
                {suggestions.map((suggestion, index) => (
                    <Button
                        key={index}
                        variant="outline"
                        className="text-left h-auto p-4 flex items-center justify-center text-center max-w-1/10"
                        onClick={() => append({ role: "user", content: suggestion })}
                    >
                        {suggestion}
                    </Button>
                ))}
            </div>
        </div>
    );
};

export { ChatContainer, ChatForm, ChatMessages, MessageList, MessageInput, PromptSuggestions };
