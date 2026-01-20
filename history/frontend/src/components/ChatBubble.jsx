export default function ChatBubble({ message, isUser }) {
  return (
    <div className={`chat-bubble ${isUser ? 'user' : 'bot'}`}>
      {!isUser && (
        <span role="img" aria-label="bot" style={{ marginRight: '0.5rem' }}>
          🤖
        </span>
      )}
      {message}
    </div>
  );
}
