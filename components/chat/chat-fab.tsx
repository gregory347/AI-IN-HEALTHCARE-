import { useState } from 'react';
import { MessageCircle, X } from 'lucide-react';
import { Button } from '../ui/button';
import { useChatStore, useAuthStore } from '../../lib/store';
import { format } from 'date-fns';
import { analyzeSymptoms } from '../../lib/ml-model';
import { toast } from 'react-hot-toast';

export function ChatFAB() {
  const { isOpen, toggleChat, messages, addMessage } = useChatStore();
  const user = useAuthStore((state) => state.user);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const hasGreeted = messages.some((msg) => msg.sender === 'system' && msg.text.includes('describe your symptoms'));

  const handleSendMessage = async (e) => {
    e.preventDefault();
    const form = e.currentTarget;
    const input = form.elements.namedItem('message');
    const message = input.value.trim();

    if (message) {
      addMessage({ text: message, sender: 'user' });
      input.value = '';

      if (!hasGreeted) {
        setTimeout(() => {
          addMessage({
            text: `hello ${user?.username || 'patient'}, describe your symptoms 😊.`,
            sender: 'system',
          });
        }, 500);
      } else {
        setIsAnalyzing(true);
        try {
          const analysis = await analyzeSymptoms(message);

          setTimeout(() => {
            addMessage({
              text: `Based on your symptoms, our doctor's analysis suggests:

Condition: ${analysis.condition}
Confidence: ${(analysis.probability * 100).toFixed(1)}%
Urgency: ${analysis.urgency.toUpperCase()}

Recommended Medications:
${analysis.medications.map((med) => `• ${med}`).join('\n')}

Other Recommendations:
${analysis.recommendations.filter((rec) => !analysis.medications.includes(rec)).map((rec) => `• ${rec}`).join('\n')}

Consultation Fees:
• Initial consultation: KSH ${analysis.consultationFees.initial}
• Follow-up visit: KSH ${analysis.consultationFees.followUp}

If symptoms persist or worsen, please schedule a consultation with one of our doctors.`,
              sender: 'system',
            });
          }, 1000);

          setTimeout(() => {
            addMessage({
              text: 'Thank you for your message. Daktari will review your case shortly.',
              sender: 'system',
            });
          }, 2000);
        } catch (error) {
          toast.error('Failed to analyze symptoms. Please try again.');
          addMessage({
            text: 'Sorry, there was an error analyzing your symptoms. Daktari will review your case directly.',
            sender: 'system',
          });
        } finally {
          setIsAnalyzing(false);
        }
      }
    }
  };

  return (
    <>
      <Button
        onClick={toggleChat}
        className="fixed bottom-4 right-4 rounded-full w-12 h-12 bg-blue-600 text-white shadow-lg z-50 transition-transform hover:scale-105"
      >
        <MessageCircle className="h-6 w-6" />
      </Button>

      {isOpen && (
        <div className="fixed bottom-20 right-4 w-[90%] max-w-sm bg-white rounded-lg shadow-xl z-50 transition-all duration-300 ease-in-out transform">
          <div className="flex items-center justify-between p-3 border-b bg-blue-600 text-white rounded-t-lg">
            <div>
              <h3 className="text-lg font-semibold">welcome to teleCure health 🩺</h3>
              <p className="text-sm text-blue-100">chat with our online doctor 👨‍⚕️ 😊</p>
            </div>
            <Button
              variant="secondary"
              size="sm"
              onClick={toggleChat}
              className="p-1 hover:bg-blue-700"
            >
              <X className="h-5 w-5" />
            </Button>
          </div>

          <div className="h-72 overflow-y-auto p-4 space-y-4 bg-gray-50">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex flex-col ${
                  message.sender === 'user' ? 'items-end' : 'items-start'
                }`}
              >
                <div
                  className={`max-w-[80%] rounded-lg p-3 whitespace-pre-wrap ${
                    message.sender === 'user'
                      ? 'bg-blue-500 text-white'
                      : message.sender === 'system'
                      ? 'bg-teal-100 text-teal-800'
                      : 'bg-green-500 text-white'
                  }`}
                >
                  {message.text}
                </div>
                <span className="text-xs text-gray-500 mt-1">
                  {format(message.timestamp, 'HH:mm')}
                </span>
              </div>
            ))}
            {isAnalyzing && (
              <div className="flex items-center justify-center space-x-2 text-gray-500">
                <div className="animate-bounce">●</div>
                <div className="animate-bounce delay-100">●</div>
                <div className="animate-bounce delay-200">●</div>
              </div>
            )}
          </div>

          <form onSubmit={handleSendMessage} className="p-3 border-t bg-white rounded-b-lg">
            <div className="flex gap-2">
              <input
                type="text"
                name="message"
                className="flex-1 rounded-md border border-gray-300 px-3 py-2 focus:border-blue-500 focus:ring-blue-500"
                placeholder="Type your message..."
                disabled={isAnalyzing}
              />
              <Button
                type="submit"
                className="bg-blue-600 hover:bg-blue-700 text-white"
                disabled={isAnalyzing}
              >
                Send
              </Button>
            </div>
          </form>
        </div>
      )}
    </>
  );
}
