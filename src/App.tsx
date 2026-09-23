import React, { useState } from "react";
import { AppHeader, NavTabId } from "./components/AppHeader";
import { HomePage } from "./pages/HomePage";
import { LearnPage } from "./pages/LearnPage";
import { PracticePage } from "./pages/PracticePage";
import { CollectPage } from "./pages/CollectPage";
import { TrainingPage } from "./pages/TrainingPage";
import { SettingsPage } from "./pages/SettingsPage";
import { AboutPage } from "./pages/AboutPage";

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<NavTabId>("recognize");
  const [targetPracticeSign, setTargetPracticeSign] = useState<string | null>(null);

  const navigateToPractice = (signLabel: string) => {
    setTargetPracticeSign(signLabel);
    setActiveTab("practice");
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "20px", alignItems: "center", width: "100%", maxWidth: "1080px", margin: "0 auto", padding: "16px" }}>
      {/* Universal Top Header with 6 Navigation Tabs */}
      <AppHeader activeTab={activeTab} onTabChange={setActiveTab} />

      {/* Main Tab Content */}
      <main style={{ width: "100%" }}>
        {activeTab === "recognize" && <HomePage onPracticeSign={navigateToPractice} onOpenFullGuide={() => setActiveTab("learn")} />}
        {activeTab === "learn" && <LearnPage onPracticeSign={navigateToPractice} />}
        {activeTab === "practice" && <PracticePage targetSign={targetPracticeSign} />}
        {activeTab === "studio" && <CollectPage />}
        {activeTab === "training" && <TrainingPage />}
        {activeTab === "settings" && <SettingsPage />}
        {activeTab === "about" && <AboutPage />}
      </main>
    </div>
  );
};

export default App;
