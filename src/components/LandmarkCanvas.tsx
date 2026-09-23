import React, { useEffect, useRef } from "react";
import { HandLandmarks } from "../types/landmarks";
import { drawHandLandmarks } from "../utils/drawLandmarks";

interface LandmarkCanvasProps {
  hands: HandLandmarks[];
  width: number;
  height: number;
  className?: string;
  style?: React.CSSProperties;
}

export const LandmarkCanvas: React.FC<LandmarkCanvasProps> = ({
  hands,
  width,
  height,
  className,
  style,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Draw frame landmarks using intrinsic pixel dimensions
    drawHandLandmarks(ctx, hands, width, height, {
      showLabels: true,
    });
  }, [hands, width, height]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className={className}
      style={style}
      aria-hidden="true"
    />
  );
};
