import { Bar, Line, Pie } from 'react-chartjs-2';
import { Chart as ChartJS } from 'chart.js/auto';

export default function GraphDisplay({ data }) {
  const renderGraph = () => {
    switch (data.type) {
      case 'bar': return <Bar data={data.data} options={data.options} />;
      case 'line': return <Line data={data.data} options={data.options} />;
      case 'pie': return <Pie data={data.data} options={data.options} />;
      default: return <Bar data={data.data} options={data.options} />;
    }
  };

  return (
    <div>
      <h3 className="font-medium mb-2 text-gray-800">{data.title}</h3>
      <div className="bg-white p-4 rounded-lg shadow">{renderGraph()}</div>
      {data.notes && (
        <p className="text-sm text-gray-600 mt-2">{data.notes}</p>
      )}
    </div>
  );
}
