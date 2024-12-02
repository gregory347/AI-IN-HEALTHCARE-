import * as tf from '@tensorflow/tfjs';

const createModel = () => {
  const model = tf.sequential();
  
  model.add(tf.layers.dense({
    inputShape: [30],
    units: 128,
    activation: 'relu'
  }));
  
  model.add(tf.layers.dense({
    units: 64,
    activation: 'relu'
  }));
  
  model.add(tf.layers.dense({
    units: 15, 
    activation: 'softmax'
  }));

  return model;
};

const symptoms = [
  'fever', 'cough', 'fatigue', 'difficulty_breathing', 'body_aches',
  'headache', 'sore_throat', 'loss_of_taste', 'nausea', 'diarrhea',
  'chills', 'rash', 'congestion', 'vomiting', 'chest_pain',
  'dizziness', 'sweating', 'muscle_weakness', 'joint_pain', 'runny_nose',
  'loss_of_smell', 'eye_pain', 'abdominal_pain', 'heart_palpitations', 'swollen_lymph_nodes',
  'weight_loss', 'appetite_loss', 'bleeding', 'shortness_of_breath', 'confusion'
];

const diseases = [
  'Common Cold', 'COVID-19', 'Flu', 'Bronchitis', 'Pneumonia',
  'Allergies', 'Asthma', 'Migraine', 'Sinus Infection', 'Tuberculosis',
  'Malaria', 'Dengue', 'Gastroenteritis', 'Chronic Fatigue Syndrome', 'Meningitis'
];

const medicationRecommendations: Record<string, string[]> = {
  'Common Cold': [
    'Paracetamol for fever and pain relief',
    'Decongestants like pseudoephedrine',
    'Cough suppressants containing dextromethorphan',
    'Antihistamines for runny nose'
  ],
  'COVID-19': [
    'Paracetamol for fever',
    'Stay hydrated and rest',
    'Follow current medical guidelines',
    'Consult doctor for specific treatments'
  ],
  'Flu': [
    'Oseltamivir (Tamiflu) if prescribed',
    'Paracetamol or ibuprofen for fever',
    'Decongestants for nasal congestion',
    'Plenty of fluids and rest'
  ],
  'Bronchitis': [
    'Expectorants to help clear mucus',
    'Cough suppressants for sleep',
    'Bronchodilators if prescribed',
    'Steam inhalation'
  ],
  'Pneumonia': [
    'Prescribed antibiotics if bacterial',
    'Pain relievers for chest pain',
    'Cough medicine',
    'Immediate medical attention required'
  ],
  'Allergies': [
    'Antihistamines (e.g., cetirizine, loratadine)',
    'Nasal corticosteroids if prescribed',
    'Decongestants for blocked nose',
    'Avoid known allergens'
  ],
  'Asthma': [
    'Inhaled bronchodilators',
    'Prescribed corticosteroids',
    'Peak flow monitoring',
    'Follow asthma action plan'
  ],
  'Migraine': [
    'Pain relievers (ibuprofen, aspirin)',
    'Anti-migraine medications if prescribed',
    'Rest in a quiet, dark room',
    'Stay hydrated'
  ],
  'Sinus Infection': [
    'Saline nasal spray',
    'Decongestants',
    'Pain relievers',
    'Antibiotics if prescribed'
  ],
  'Tuberculosis': [
    'Prescribed TB medications only',
    'Complete full course of treatment',
    'Regular medical monitoring',
    'Immediate medical attention required'
  ],
  'Malaria': [
    'Prescribed antimalarial medications',
    'Fever reducers',
    'Immediate medical attention required',
    'Complete prescribed course'
  ],
  'Dengue': [
    'Paracetamol for fever (avoid aspirin)',
    'Plenty of fluids',
    'Rest and monitoring',
    'Immediate medical attention required'
  ],
  'Gastroenteritis': [
    'Oral rehydration solutions',
    'Anti-diarrheal medication if needed',
    'Bland diet (BRAT)',
    'Probiotics may help'
  ],
  'Chronic Fatigue Syndrome': [
    'Pain relievers as needed',
    'Sleep medications if prescribed',
    'Antidepressants if prescribed',
    'Professional medical supervision required'
  ],
  'Meningitis': [
    'Emergency medical attention required',
    'Prescribed antibiotics if bacterial',
    'Pain relief medication',
    'Hospital treatment necessary'
  ]
};

const consultationFees: Record<string, { initial: number; followUp: number }> = {
  'Common Cold': { initial: 300, followUp: 200 },
  'COVID-19': { initial: 2000, followUp: 1000 },
  'Flu': { initial: 400, followUp: 250 },
  'Bronchitis': { initial: 450, followUp: 300 },
  'Pneumonia': { initial: 600, followUp: 400 },
  'Allergies': { initial: 350, followUp: 250 },
  'Asthma': { initial: 500, followUp: 350 },
  'Migraine': { initial: 400, followUp: 300 },
  'Sinus Infection': { initial: 350, followUp: 250 },
  'Tuberculosis': { initial: 5000, followUp: 2500 },
  'Malaria': { initial: 550, followUp: 400 },
  'Dengue': { initial: 6000, followUp: 4500 },
  'Gastroenteritis': { initial: 4500, followUp: 3000 },
  'Chronic Fatigue Syndrome': { initial: 6500, followUp: 4000 },
  'Meningitis': { initial: 8000, followUp: 6000 }
};

const severityWeights: { [key: string]: number } = {
  fever: 0.6, cough: 0.4, fatigue: 0.3, difficulty_breathing: 0.8, body_aches: 0.3,
  headache: 0.2, sore_throat: 0.2, loss_of_taste: 0.4, nausea: 0.3, diarrhea: 0.3,
  chills: 0.4, rash: 0.3, congestion: 0.2, vomiting: 0.5, chest_pain: 0.7,
  dizziness: 0.3, sweating: 0.3, muscle_weakness: 0.4, joint_pain: 0.3, runny_nose: 0.2,
  loss_of_smell: 0.4, eye_pain: 0.2, abdominal_pain: 0.5, heart_palpitations: 0.7, swollen_lymph_nodes: 0.3,
  weight_loss: 0.6, appetite_loss: 0.4, bleeding: 0.8, shortness_of_breath: 0.8, confusion: 0.7
};

let model: tf.LayersModel | null = null;

async function initModel() {
  if (!model) {
    try {
      model = createModel();
      model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
      });
      
      const weights = model.getWeights();
      const initializedWeights = weights.map(w => 
        tf.randomNormal(w.shape, 0, 0.1)
      );
      model.setWeights(initializedWeights);
    } catch (error) {
      console.error('Error initializing model:', error);
      throw new Error('Failed to initialize symptom analysis model');
    }
  }
  return model;
}

function extractSymptoms(text: string): number[] {
  const lowercaseText = text.toLowerCase();
  return symptoms.map(symptom => 
    lowercaseText.includes(symptom.toLowerCase().replace('_', ' ')) ? 1 : 0
  );
}

function calculateSeverity(symptomVector: number[]): number {
  return symptoms.reduce((score, symptom, index) => 
    score + (symptomVector[index] * (severityWeights[symptom] || 0)), 0);
}

function generateRecommendations(condition: string, urgency: string): string[] {
  const medications = medicationRecommendations[condition] || [];
  const fees = consultationFees[condition];
  
  const baseRecommendations = [
    'Rest and stay hydrated',
    'Monitor your symptoms',
    ...medications
  ];

  const urgencyRecommendations: Record<string, string[]> = {
    high: [
      'Seek immediate medical attention',
      'Schedule an urgent consultation',
      `Initial consultation fee: KSH${fees.initial}`,
      'Consider emergency care if symptoms worsen'
    ],
    medium: [
      'Schedule a follow-up appointment',
      'Begin prescribed treatment plan',
      `Initial consultation fee: KSH${fees.initial}`,
      `Follow-up consultation fee: KSH${fees.followUp}`,
      'Contact your doctor if symptoms persist'
    ],
    low: [
      'Continue normal activities with caution',
      'Use over-the-counter medications as listed above',
      'If symptoms persist beyond 5-7 days:',
      `  - Schedule a consultation (Fee: KSH${fees.initial})`,
      'Practice preventive measures'
    ]
  };

  return [...baseRecommendations, ...(urgencyRecommendations[urgency] || [])];
}

export interface AnalysisResult {
  condition: string;
  probability: number;
  recommendations: string[];
  urgency: 'low' | 'medium' | 'high';
  medications: string[];
  consultationFees: {
    initial: number;
    followUp: number;
  };
}

export async function analyzeSymptoms(symptomText: string): Promise<AnalysisResult> {
  try {
    const model = await initModel();
    const symptomVector = extractSymptoms(symptomText);
    const inputTensor = tf.tensor2d([symptomVector], [1, symptoms.length]);
    const prediction = model.predict(inputTensor) as tf.Tensor;
    const probabilities = await prediction.data();
    const maxIndex = Array.from(probabilities).indexOf(Math.max(...Array.from(probabilities)));
    const condition = diseases[maxIndex];
    const probability = probabilities[maxIndex];
    const severityScore = calculateSeverity(symptomVector);
    let urgency: 'low' | 'medium' | 'high';

    if (severityScore > 0.6) urgency = 'high';
    else if (severityScore > 0.3) urgency = 'medium';
    else urgency = 'low';

    const recommendations = generateRecommendations(condition, urgency);
    const medications = medicationRecommendations[condition] || [];
    const fees = consultationFees[condition];

    tf.dispose([inputTensor, prediction]);

    return {
      condition,
      probability,
      recommendations,
      urgency,
      medications,
      consultationFees: fees
    };
  } catch (error) {
    console.error('Error analyzing symptoms:', error);
    throw new Error('Failed to analyze symptoms');
  }
}