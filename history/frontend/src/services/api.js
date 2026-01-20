export const sendMessage = async (message) => {
  try {
    const response = await fetch('http://localhost:5000/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ message })
    });

    if (!response.ok) {
      throw new Error('Failed to fetch response');
    }

    const data = await response.json();

    return {
      text: data.text || "No response received.",
      graph: data.graph || null
    };
  } catch (error) {
    console.error('API Error:', error);
    return {
      text: "Server error! 😞 Try again later.",
      graph: null
    };
  }
};
