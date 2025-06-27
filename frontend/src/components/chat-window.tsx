import React, { useState, useRef } from "react";
import {
  ChatContainer,
  ChatForm,
  MessageInput,
  PromptSuggestions,
  ChatMessages,
  MessageList,
} from "@/components/ui/custom";
import { Loader2 } from "lucide-react";

type Message = {
  id: string;
  role: string;
  content: string;
};

type ToolStep = {
  tool_name: string;
  args: Record<string, any>;
};

type Plan = {
  steps: ToolStep[];
};

const PlanDisplay = ({ plan, toolStatus }: { plan: Plan; toolStatus: Record<string, string> }) => (
  <div className="bg-gray-800 p-4 rounded-lg my-2 text-sm">
    <h3 className="font-bold mb-2 text-base">Execution Plan:</h3>
    <ul className="space-y-2">
      {plan.steps.map((step, index) => (
        <li key={index} className="flex items-center">
          {toolStatus[index] === 'running' && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
          {toolStatus[index] === 'finished' && <span className="mr-2 text-green-500">✅</span>}
          {!toolStatus[index] && <span className="mr-2"></span>}
          <span className="font-mono">{step.tool_name}({JSON.stringify(step.args)})</span>
        </li>
      ))}
    </ul>
  </div>
);

export const ChatWindow = () => {
    const [messages, setMessages] = useState<Array<Message>>([]);
    const [input, setInput] = useState("");
    const [isGenerating, setIsGenerating] = useState(false);
    const [plan, setPlan] = useState<Plan | null>(null);
    const [toolStatus, setToolStatus] = useState<Record<number, string>>({});

    const abortControllerRef = useRef<AbortController | null>(null);
    const API_URL_BASE = import.meta.env.VITE_API_BASE_URL;
    const isEmpty = messages.length === 0;

    const getCompletion = async (queryMessage: Message) => {
        setMessages(prev => [...prev, queryMessage]);
        setIsGenerating(true);
        abortControllerRef.current = new AbortController();
        setPlan(null);
        setToolStatus({});

        const payload = { messages: [...messages, queryMessage] };

        try {
            const response = await fetch(API_URL_BASE + "/api/query", {
                method: "POST",
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
                signal: abortControllerRef.current.signal,
            });

            if (!response.body) return;
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let assistantMessage: Message = { id: (messages.length + 2).toString(), role: 'assistant', content: '' };
            let assistantMessageAdded = false;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value);
                const eventLines = chunk.split('data: ').filter(line => line.trim());

                for (const line of eventLines) {
                    try {
                        const event = JSON.parse(line);
                        if (event.type === 'plan') {
                            setPlan(event.data);
                        } else if (event.type === 'tools_start') {
                            setToolStatus(prev => {
                                const newStatus: Record<number, string> = {};
                                plan?.steps.forEach((_, index) => newStatus[index] = 'running');
                                return newStatus;
                            });
                        } else if (event.type === 'tools_end') {
                             setToolStatus(prev => {
                                const newStatus: Record<number, string> = {};
                                plan?.steps.forEach((_, index) => newStatus[index] = 'finished');
                                return newStatus;
                            });
                        } else if (event.type === 'final_answer') {
                            if (!assistantMessageAdded) {
                                setMessages(prev => [...prev, assistantMessage]);
                                assistantMessageAdded = true;
                            }
                            assistantMessage.content = event.data;
                            setMessages(prev => [...prev.slice(0, -1), { ...assistantMessage }]);
                        }
                    } catch (e) {
                        // console.error("Error parsing stream chunk", e, "line:", line);
                    }
                }
            }
        } catch (error) {
            // ... (error handling)
        } finally {
            setIsGenerating(false);
            abortControllerRef.current = null;
        }
    };

    const appendMessage = (message: { role: string; content: string }) => {
        const newMessage = { id: messages.length.toString(), ...message };
        getCompletion(newMessage);
    };

    const submitMessage = (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim()) return;
        appendMessage({ role: "user", content: input });
        setInput("");
    };
    
    const stop = () => {
      // ...
    };

    return (
        <ChatContainer className="h-screen max-h-full flex flex-col gap-6 justify-center w-full sm:w-3/5">
            {isEmpty ? (
                <h1 className="text-3xl font-bold text-center">Caltech Catalog Agent</h1>
            ) : (
                <h1 className="text-xl font-bold text-center">Caltech Catalog Agent</h1>
            )}

            {isEmpty && (
                 <div className="space-y-6">
                    <PromptSuggestions
                        label="Don't know what to ask? Try these prompts!"
                        append={appendMessage}
                        suggestions={[
                            "Tell me about the study abroad programs!",
                            "I like philosophizing time travel. Are there any classes about this?",
                            "What do students think about CS 1?",
                        ]}
                    />
                </div>
            )}

            {!isEmpty && (
                <div className="space-y-2 max-h-3/4 overflow-y-auto px-4">
                    <ChatMessages messages={messages}>
                        <MessageList messages={messages} />
                    </ChatMessages>
                    {plan && <PlanDisplay plan={plan} toolStatus={toolStatus} />}
                </div>
            )}

            <ChatForm
                className="mt-auto"
                handleSubmit={submitMessage}
                stop={stop}
            >
                {() => (
                    <MessageInput
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        stop={stop}
                        isGenerating={isGenerating}
                    />
                )}
            </ChatForm>
        </ChatContainer>
    );
};