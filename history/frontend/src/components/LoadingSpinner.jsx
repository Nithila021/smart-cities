import { useEffect, useState } from 'react';

export default function LoadingSpinner() {
  const [dots, setDots] = useState('');

  useEffect(() => {
    const interval = setInterval(() => {
      setDots((prev) => (prev.length < 3 ? prev + '.' : ''));
    }, 400);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="chat-bubble bot loading-spinner">
      <span role="img" aria-label="bot">🤖</span>
      Typing{dots}
    </div>
  );
}
