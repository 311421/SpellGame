using System.Collections.Generic;
using UnityEngine;

namespace SpellDrawing.CV
{
    /// <summary>Converts a drawn stroke into a small square grayscale bitmap (aspect-preserving,
    /// unlike DollarOneRecognizer.Normalize's per-axis scale) — the shared first step for any
    /// image-based technique (shape descriptors, a trained CNN).</summary>
    public static class StrokeRasterizer
    {
        /// <summary>Single-stroke convenience wrapper around the multi-stroke overload below.</summary>
        public static Texture2D Rasterize(
            IReadOnlyList<Vector2> points, int size = 64, float strokeThicknessPx = 3f, float paddingFraction = 0.12f)
        {
            return Rasterize(new List<List<Vector2>> { new List<Vector2>(points) }, size, strokeThicknessPx, paddingFraction);
        }

        /// <summary>Rasterizes a multi-stroke gesture (size x size, grayscale-on-black RGBA32, white =
        /// ink). Each stroke is its own polyline — never connects one stroke's end to the next
        /// stroke's start.</summary>
        public static Texture2D Rasterize(
            IReadOnlyList<List<Vector2>> strokes, int size = 64, float strokeThicknessPx = 2f, float paddingFraction = 0.12f)
        {
            var fittedStrokes = FitStrokesToUnitSquare(strokes, paddingFraction);
            float thicknessNorm = strokeThicknessPx / size;

            var pixels = new Color32[size * size];
            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    Vector2 pixelCenter = new Vector2((x + 0.5f) / size, (y + 0.5f) / size);

                    float dist = float.MaxValue;
                    foreach (var stroke in fittedStrokes)
                    {
                        float d = DistanceToPolyline(pixelCenter, stroke);
                        if (d < dist) dist = d;
                    }

                    float t = Mathf.Clamp01((dist - thicknessNorm * 0.5f) / (thicknessNorm * 0.5f));
                    byte v = (byte)(255 * (1f - t));
                    pixels[y * size + x] = new Color32(v, v, v, 255);
                }
            }

            var texture = new Texture2D(size, size, TextureFormat.RGBA32, false)
            {
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp
            };
            texture.SetPixels32(pixels);
            texture.Apply();
            return texture;
        }

        /// <summary>Maps every stroke into [0,1]x[0,1] texture space using one shared transform
        /// so strokes keep their position/size relative to each other.</summary>
        private static List<List<Vector2>> FitStrokesToUnitSquare(
            IReadOnlyList<List<Vector2>> strokes, float paddingFraction)
        {
            Vector2 min = strokes[0][0];
            Vector2 max = strokes[0][0];
            foreach (var stroke in strokes)
            {
                foreach (var p in stroke)
                {
                    min = Vector2.Min(min, p);
                    max = Vector2.Max(max, p);
                }
            }

            Vector2 center = (min + max) * 0.5f;
            float maxDim = Mathf.Max(max.x - min.x, max.y - min.y, 0.0001f);
            float usableFraction = 1f - 2f * paddingFraction;
            float scale = usableFraction / maxDim;

            var result = new List<List<Vector2>>(strokes.Count);
            foreach (var stroke in strokes)
            {
                var fitted = new List<Vector2>(stroke.Count);
                foreach (var p in stroke) fitted.Add((p - center) * scale + new Vector2(0.5f, 0.5f));
                result.Add(fitted);
            }
            return result;
        }

        private static float DistanceToPolyline(Vector2 p, List<Vector2> polyline)
        {
            float minDist = float.MaxValue;
            for (int i = 0; i < polyline.Count - 1; i++)
            {
                float d = DistancePointToSegment(p, polyline[i], polyline[i + 1]);
                if (d < minDist) minDist = d;
            }
            return minDist;
        }

        private static float DistancePointToSegment(Vector2 p, Vector2 a, Vector2 b)
        {
            Vector2 ab = b - a;
            float lengthSq = ab.sqrMagnitude;
            if (lengthSq < 1e-8f) return Vector2.Distance(p, a);

            float t = Mathf.Clamp01(Vector2.Dot(p - a, ab) / lengthSq);
            Vector2 projection = a + t * ab;
            return Vector2.Distance(p, projection);
        }
    }
}
