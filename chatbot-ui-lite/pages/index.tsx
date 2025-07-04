import { Chat } from "@/components/Chat/Chat";
import { Footer } from "@/components/Layout/Footer";
import { Navbar } from "@/components/Layout/Navbar";
import { Message } from "@/types";
import { clearSessionId } from "@/utils/session";
import { clear } from "console";
import Head from "next/head";
import { useEffect, useRef, useState } from "react";

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    clearSessionId();
  }, []);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleSend = async (message: Message, session: string) => {
    const question = message.content;
    setLoading(true);
  
    // Add user's message to UI
    setMessages((messages) => [...messages, message]);
  
    try {
      const res = await fetch('http://54.89.163.204:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: question, session_id: session }),
      });
  
      if (!res.ok) {
        throw new Error(`API error: ${res.status} ${res.statusText}`);
      }
  
      const { response } = await res.json();
  
      setMessages((messages) => [
        ...messages,
        {
          role: "assistant",
          content: response
        }
      ]);
    } catch (err) {
      console.error(err);
      setMessages((messages) => [
        ...messages,
        {
          role: "assistant",
          content: "Something went wrong. Please try again."
        }
      ]);
    } finally {
      setLoading(false);
    }
  };
  

  const handleReset = () => {
    setMessages([
      {
        role: "assistant",
        content: `Hi! Ask a question about Christian!`
      }
    ]);
    clearSessionId();
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    setMessages([
      {
        role: "assistant",
        content: `Hi! Ask a question about Christian!`
      }
    ]);
  }, []);

  return (
    <>
      <Head>
        <title>Christian's Chatbot</title>
        <meta
          name="description"
          content="A simple chatbot starter kit for OpenAI's chat model using Next.js, TypeScript, and Tailwind CSS."
        />
        <meta
          name="viewport"
          content="width=device-width, initial-scale=1"
        />
        <link
          rel="icon"
          href="/favicon.ico"
        />
      </Head>

      <div className="flex flex-col h-screen">
        <Navbar />

        <div className="flex-1 overflow-auto sm:px-10 pb-4 sm:pb-10">
          <div className="max-w-[800px] mx-auto mt-4 sm:mt-12">
            <Chat
              messages={messages}
              loading={loading}
              onSend={handleSend}
              onReset={handleReset}
            />
            <div ref={messagesEndRef} />
          </div>
        </div>
        <Footer />
      </div>
    </>
  );
}
