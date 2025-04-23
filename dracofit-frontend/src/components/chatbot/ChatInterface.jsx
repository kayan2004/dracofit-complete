import React, { useState, useEffect, useRef, useCallback } from "react";
import ChatMessage from "./ChatMessage";
import ChatInput from "./ChatInput";
import chatbotService from "../../services/chatbotService";

// --- Define a key for localStorage ---
const CHAT_HISTORY_KEY = "dracofit-chat-history-v2";

const ChatInterface = () => {
  // --- Load initial state from localStorage ---
  const [messages, setMessages] = useState(() => {
    try {
      const savedMessages = localStorage.getItem(CHAT_HISTORY_KEY);
      if (savedMessages) {
        const parsedMessages = JSON.parse(savedMessages);
        // Ensure timestamps are Date objects and other flags exist
        return parsedMessages.map((msg) => ({
          ...msg,
          timestamp: msg.timestamp ? new Date(msg.timestamp) : null,
          isLoading: msg.isLoading || false, // Ensure flags exist
          isError: msg.isError || false,
        }));
      }
    } catch (error) {
      console.error("Failed to load chat history from localStorage:", error);
    }
    // Default initial message if nothing is saved or loading failed
    return [
      {
        id: 1,
        text: "Hello! I'm DracoBot, your fitness assistant. How can I help you today?",
        sender: "bot",
        timestamp: new Date(),
        isLoading: false,
        isError: false,
      },
    ];
  });

  const [loading, setLoading] = useState(false);
  const [modelStatus, setModelStatus] = useState({
    status: "unknown", // Keep initial state as unknown
    lastChecked: null,
    details: null,
    error: null,
  });
  const [apiError, setApiError] = useState(null);
  const messagesEndRef = useRef(null);
  const eventSourceRef = useRef(null);

  // --- Save messages to localStorage whenever they change ---
  useEffect(() => {
    try {
      localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(messages));
    } catch (error) {
      console.error("Failed to save chat history to localStorage:", error);
    }
    // scroll to bottom when messages update
    scrollToBottom();
  }, [messages]);

  // Check model health on component mount
  useEffect(() => {
    checkModelStatus();
    const intervalId = setInterval(checkModelStatus, 60000);
    return () => clearInterval(intervalId);
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const checkModelStatus = async () => {
    try {
      const health = await chatbotService.checkHealth();
      setModelStatus({
        status: health.online ? "online" : "offline",
        lastChecked: new Date(),
        details: health.details, // Store details if returned
        error: null,
      });
    } catch (error) {
      console.error("Health check failed:", error); // Log the actual error
      setModelStatus({
        status: "error", // Use 'error' status
        lastChecked: new Date(),
        error: error.message, // Store error message
        details: null,
      });
    }
  };

  // --- Streaming Message Handler (handleSendMessage) ---
  const handleSendMessage = useCallback(
    async (text) => {
      if (!text.trim() || loading || modelStatus.status !== "online") return; // Check status before sending

      setLoading(true);
      setApiError(null);

      const newUserMessage = {
        id: `user-${Date.now()}`,
        text,
        sender: "user",
        timestamp: new Date(),
        isLoading: false,
        isError: false,
      };
      setMessages((prevMessages) => [...prevMessages, newUserMessage]);

      const botMessageId = `bot-${Date.now()}`;
      setMessages((prevMessages) => [
        ...prevMessages,
        {
          id: botMessageId,
          text: "",
          sender: "bot",
          timestamp: new Date(),
          isLoading: true, // Mark as loading
          isError: false,
        },
      ]);

      try {
        const apiUrl = "http://localhost:5000/chat";

        const response = await fetch(apiUrl, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "text/event-stream",
          },
          body: JSON.stringify({ message: text }),
          credentials: "include",
        });

        if (!response.ok || !response.body) {
          const errorText = await response
            .text()
            .catch(() => `HTTP error ${response.status}`);
          throw new Error(
            `Failed to connect to stream: ${response.status} ${errorText}`
          );
        }

        const reader = response.body
          .pipeThrough(new TextDecoderStream())
          .getReader();
        let buffer = "";
        let receivedText = ""; // Accumulate text for the current bot message

        while (true) {
          const { value, done } = await reader.read();
          if (done) {
            // Stream finished, mark the bot message as not loading
            setMessages((prevMessages) =>
              prevMessages.map((msg) =>
                msg.id === botMessageId ? { ...msg, isLoading: false } : msg
              )
            );
            break;
          }

          buffer += value;
          let boundary = buffer.indexOf("\n\n");
          while (boundary !== -1) {
            const messageLine = buffer.substring(0, boundary);
            buffer = buffer.substring(boundary + 2);

            if (messageLine.startsWith("data: ")) {
              const jsonData = messageLine.substring(6);
              try {
                const parsedData = JSON.parse(jsonData);

                if (parsedData.status === "streaming" && parsedData.chunk) {
                  receivedText += parsedData.chunk;
                  setMessages((prevMessages) =>
                    prevMessages.map((msg) =>
                      msg.id === botMessageId
                        ? { ...msg, text: receivedText, isLoading: true } // Update text, keep loading
                        : msg
                    )
                  );
                } else if (parsedData.status === "success") {
                  // Success event might contain full response
                  setMessages((prevMessages) =>
                    prevMessages.map((msg) =>
                      msg.id === botMessageId
                        ? {
                            ...msg,
                            text: parsedData.full_response || receivedText,
                            isLoading: false,
                          } // Set final text, stop loading
                        : msg
                    )
                  );
                  // Stream should end via 'done' flag shortly after
                } else if (
                  parsedData.status === "error" ||
                  parsedData.status === "aborted"
                ) {
                  console.error("Stream error/abort:", parsedData.message);
                  setApiError(`Stream Error: ${parsedData.message}`);
                  setMessages((prevMessages) =>
                    prevMessages.map((msg) =>
                      msg.id === botMessageId
                        ? {
                            ...msg,
                            isLoading: false,
                            isError: true,
                            text:
                              msg.text +
                              ` [${parsedData.status}: ${parsedData.message}]`,
                          }
                        : msg
                    )
                  );
                  return; // Stop processing on error/abort
                }
              } catch (e) {
                console.error("Failed to parse SSE data:", jsonData, e);
              }
            }
            boundary = buffer.indexOf("\n\n");
          }
        } // end while reader
      } catch (error) {
        console.error("Error sending/streaming message:", error);
        setApiError(`Error: ${error.message}`);
        setMessages((prevMessages) =>
          prevMessages.map((msg) =>
            msg.id === botMessageId
              ? {
                  ...msg,
                  isLoading: false,
                  isError: true,
                  text: `[Failed to get response: ${error.message}]`,
                }
              : msg
          )
        );
      } finally {
        setLoading(false);
        // Ensure loading is false if loop finishes unexpectedly
        setMessages((prevMessages) =>
          prevMessages.map((msg) =>
            msg.id === botMessageId ? { ...msg, isLoading: false } : msg
          )
        );
      }
    },
    [loading, modelStatus.status] // Add modelStatus.status dependency
  );

  // --- getStatusBadge function (Unchanged) ---
  const getStatusBadge = () => {
    if (modelStatus.status === "unknown") {
      // Optionally show a checking state or return null
      return (
        <div className="bg-gray-700 text-white text-xs px-3 py-1 rounded-full mb-4 animate-pulse">
          Checking status...
        </div>
      );
    }

    if (modelStatus.status === "offline") {
      return (
        <div className="bg-red-900 text-white text-xs px-3 py-1 rounded-full mb-4">
          ⚠️ DracoBot is offline
        </div>
      );
    }

    if (modelStatus.status === "error") {
      // Display specific error or generic message
      return (
        <div
          className="bg-yellow-900 text-white text-xs px-3 py-1 rounded-full mb-4"
          title={modelStatus.error || "Connection error"}
        >
          ⚠️ Connection issues
        </div>
      );
    }

    // Default to online if status is 'online'
    return (
      <div className="bg-green-900 text-white text-xs px-3 py-1 rounded-full mb-4">
        ✓ DracoBot is online
      </div>
    );
  };
  // --- End of getStatusBadge function ---

  return (
    <div className="flex flex-col h-[100%]">
      {" "}
      {/* Ensure parent has height */}
      {/* --- Header Section (Unchanged) --- */}
      <div className="flex justify-between items-center mb-4 px-4 pt-4">
        {" "}
        {/* Added padding */}
        <h2 className="text-xl font-bold text-white">DracoBot Chat</h2>
        {getStatusBadge()}
      </div>
      {/* --- End of Header Section --- */}
      {/* Chat Messages Area */}
      <div className="flex-1 overflow-y-auto pb-4 px-4 space-y-4">
        {" "}
        {/* Added padding and space-y */}
        {messages.map((message) => (
          <ChatMessage
            key={message.id}
            message={message}
            // isLoading prop is redundant if passed in message object, but kept for compatibility
            isLoading={message.isLoading}
          />
        ))}
        <div ref={messagesEndRef} /> {/* For scrolling */}
      </div>
      {/* Error Display Area */}
      {apiError && (
        <div className="p-2 text-center text-red-400 bg-red-900/50 text-sm mx-4 mb-2 rounded">
          {" "}
          {/* Added styling */}
          {apiError}
        </div>
      )}
      {/* Chat Input Area */}
      <div className="p-4 border-t border-gray-700">
        <ChatInput
          onSendMessage={handleSendMessage}
          isLoading={loading} // Overall loading state for input
          isDisabled={modelStatus.status !== "online"} // Disable input if not online
        />
      </div>
    </div>
  );
};

export default ChatInterface;
