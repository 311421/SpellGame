using System.Collections.Generic;
using UnityEngine;

namespace SpellDrawing
{
    /// <summary>$1 Unistroke Recognizer (Wobbrock, Wilson &amp; Li, 2007): normalizes a raw stroke
    /// and matches it against recorded SpellTemplate gestures.</summary>
    public static class DollarOneRecognizer
    {
        public const int ResampleCount = 64;
        public const float SquareSize = 250f;

        private const float AngleRange = 45f * Mathf.Deg2Rad;
        private const float AnglePrecision = 2f * Mathf.Deg2Rad;
        private const float CornerPenaltyWeight = 0.6f;
        private static readonly float PhiRatio = 0.5f * (-1f + Mathf.Sqrt(5f));

        public struct Result
        {
            public SpellTemplate template;
            public float score;
            public List<(SpellTemplate template, float score)> ranked;
        }

        public static List<Vector2> Normalize(IReadOnlyList<Vector2> rawPoints)
        {
            var points = Resample(rawPoints, ResampleCount);
            float radians = IndicativeAngle(points);
            points = RotateBy(points, -radians);
            points = ScaleToSquare(points, SquareSize);
            points = TranslateToOrigin(points);
            return points;
        }

        /// <param name="fullRotationSearch">False (default, live gameplay) searches only +/-45deg
        /// around the indicative angle. True does a full 360deg search — needed for rotationally
        /// symmetric shapes, where the indicative angle is arbitrary and a narrow window would only
        /// sometimes land close enough by luck; costs more, so only the offline dataset generator uses it.</param>
        public static Result Recognize(
            IReadOnlyList<Vector2> rawPoints, IReadOnlyList<SpellTemplate> templates, bool fullRotationSearch = false)
        {
            // Normalize both drawn order and its reverse, keep whichever aligns better per template —
            // makes matching indifferent to which way the player traced the gesture.
            var forward = Normalize(rawPoints);
            var reversedRaw = new List<Vector2>(rawPoints);
            reversedRaw.Reverse();
            var backward = Normalize(reversedRaw);

            float halfDiagonal = 0.5f * Mathf.Sqrt(SquareSize * SquareSize + SquareSize * SquareSize);
            // Corner score is unaffected by reversal (turning-angle magnitude only), so one suffices.
            float candidateCornerScore = ComputeCornerScore(forward);

            var ranked = new List<(SpellTemplate template, float score)>();

            foreach (var template in templates)
            {
                if (template == null || template.NormalizedPoints.Count != ResampleCount) continue;

                float distanceForward = fullRotationSearch
                    ? FindBestRotationDistance(forward, template.NormalizedPoints)
                    : DistanceAtBestAngle(forward, template.NormalizedPoints, -AngleRange, AngleRange, AnglePrecision);
                float distanceBackward = fullRotationSearch
                    ? FindBestRotationDistance(backward, template.NormalizedPoints)
                    : DistanceAtBestAngle(backward, template.NormalizedPoints, -AngleRange, AngleRange, AnglePrecision);
                float distance = Mathf.Min(distanceForward, distanceBackward);

                float pathScore = Mathf.Clamp01(1f - distance / halfDiagonal);

                float cornerPenalty = Mathf.Abs(candidateCornerScore - template.CornerScore);
                float score = pathScore * (1f - CornerPenaltyWeight * cornerPenalty);

                ranked.Add((template, score));
            }

            ranked.Sort((a, b) => b.score.CompareTo(a.score));

            var result = new Result { ranked = ranked };
            if (ranked.Count > 0)
            {
                result.template = ranked[0].template;
                result.score = ranked[0].score;
            }
            return result;
        }

        /// <summary>0 (round) to 1 (sharp corners), from the average of the sharpest turning angles.</summary>
        public static float ComputeCornerScore(IReadOnlyList<Vector2> normalizedPoints, int topK = 3)
        {
            int n = normalizedPoints.Count;
            if (n < 3) return 0f;

            var turningAngles = new List<float>(n - 2);
            for (int i = 1; i < n - 1; i++)
            {
                Vector2 prevDir = normalizedPoints[i] - normalizedPoints[i - 1];
                Vector2 nextDir = normalizedPoints[i + 1] - normalizedPoints[i];
                if (prevDir.sqrMagnitude < 0.0001f || nextDir.sqrMagnitude < 0.0001f) continue;
                turningAngles.Add(Vector2.Angle(prevDir, nextDir) * Mathf.Deg2Rad);
            }
            if (turningAngles.Count == 0) return 0f;

            turningAngles.Sort((a, b) => b.CompareTo(a));
            int count = Mathf.Min(topK, turningAngles.Count);
            float sum = 0f;
            for (int i = 0; i < count; i++) sum += turningAngles[i];
            float averageTopAngle = sum / count;

            return Mathf.Clamp01(averageTopAngle / (Mathf.PI * 0.5f));
        }

        /// <summary>Long-side/short-side ratio (always &gt;= 1) after derotating but BEFORE
        /// ScaleToSquare's non-uniform scale — the one piece of shape info Normalize() discards.</summary>
        public static float ComputeAspectRatio(IReadOnlyList<Vector2> rawPoints)
        {
            var points = Resample(rawPoints, ResampleCount);
            float radians = IndicativeAngle(points);
            points = RotateBy(points, -radians);

            Rect box = BoundingBox(points);
            float w = Mathf.Max(box.width, 0.0001f);
            float h = Mathf.Max(box.height, 0.0001f);
            return Mathf.Max(w, h) / Mathf.Min(w, h);
        }

        private static List<Vector2> Resample(IReadOnlyList<Vector2> points, int n)
        {
            float interval = PathLength(points) / (n - 1);
            if (interval <= 0f) interval = 0.0001f;

            float accumulated = 0f;
            var src = new List<Vector2>(points);
            var result = new List<Vector2> { src[0] };

            for (int i = 1; i < src.Count; i++)
            {
                float d = Vector2.Distance(src[i - 1], src[i]);
                if (accumulated + d >= interval)
                {
                    float t = d > 0f ? (interval - accumulated) / d : 0f;
                    Vector2 q = Vector2.Lerp(src[i - 1], src[i], t);
                    result.Add(q);
                    src.Insert(i, q);
                    accumulated = 0f;
                }
                else
                {
                    accumulated += d;
                }
            }

            while (result.Count < n) result.Add(src[src.Count - 1]);
            if (result.Count > n) result.RemoveRange(n, result.Count - n);
            return result;
        }

        private static float PathLength(IReadOnlyList<Vector2> points)
        {
            float length = 0f;
            for (int i = 1; i < points.Count; i++) length += Vector2.Distance(points[i - 1], points[i]);
            return length;
        }

        private static float IndicativeAngle(List<Vector2> points)
        {
            Vector2 centroid = Centroid(points);
            return Mathf.Atan2(centroid.y - points[0].y, centroid.x - points[0].x);
        }

        private static List<Vector2> RotateBy(List<Vector2> points, float radians)
        {
            Vector2 centroid = Centroid(points);
            float cos = Mathf.Cos(radians);
            float sin = Mathf.Sin(radians);

            var result = new List<Vector2>(points.Count);
            foreach (var p in points)
            {
                float dx = p.x - centroid.x;
                float dy = p.y - centroid.y;
                result.Add(new Vector2(
                    dx * cos - dy * sin + centroid.x,
                    dx * sin + dy * cos + centroid.y));
            }
            return result;
        }

        private static List<Vector2> ScaleToSquare(List<Vector2> points, float size)
        {
            Rect box = BoundingBox(points);
            var result = new List<Vector2>(points.Count);
            foreach (var p in points)
            {
                result.Add(new Vector2(
                    box.width > 0.0001f ? p.x * (size / box.width) : p.x,
                    box.height > 0.0001f ? p.y * (size / box.height) : p.y));
            }
            return result;
        }

        private static List<Vector2> TranslateToOrigin(List<Vector2> points)
        {
            Vector2 centroid = Centroid(points);
            var result = new List<Vector2>(points.Count);
            foreach (var p in points) result.Add(new Vector2(p.x - centroid.x, p.y - centroid.y));
            return result;
        }

        private static Vector2 Centroid(IReadOnlyList<Vector2> points)
        {
            float x = 0f, y = 0f;
            foreach (var p in points) { x += p.x; y += p.y; }
            return new Vector2(x / points.Count, y / points.Count);
        }

        private static Rect BoundingBox(IReadOnlyList<Vector2> points)
        {
            float minX = float.MaxValue, minY = float.MaxValue, maxX = float.MinValue, maxY = float.MinValue;
            foreach (var p in points)
            {
                minX = Mathf.Min(minX, p.x); minY = Mathf.Min(minY, p.y);
                maxX = Mathf.Max(maxX, p.x); maxY = Mathf.Max(maxY, p.y);
            }
            return new Rect(minX, minY, maxX - minX, maxY - minY);
        }

        private static float PathDistance(IReadOnlyList<Vector2> a, IReadOnlyList<Vector2> b)
        {
            float d = 0f;
            int count = Mathf.Min(a.Count, b.Count);
            for (int i = 0; i < count; i++) d += Vector2.Distance(a[i], b[i]);
            return d / count;
        }

        /// <summary>Golden-section search for the rotation angle that minimizes distance to the template.</summary>
        private static float DistanceAtBestAngle(
            List<Vector2> points, IReadOnlyList<Vector2> template, float a, float b, float threshold)
        {
            float x1 = PhiRatio * a + (1f - PhiRatio) * b;
            float f1 = DistanceAtAngle(points, template, x1);
            float x2 = (1f - PhiRatio) * a + PhiRatio * b;
            float f2 = DistanceAtAngle(points, template, x2);

            while (Mathf.Abs(b - a) > threshold)
            {
                if (f1 < f2)
                {
                    b = x2; x2 = x1; f2 = f1;
                    x1 = PhiRatio * a + (1f - PhiRatio) * b;
                    f1 = DistanceAtAngle(points, template, x1);
                }
                else
                {
                    a = x1; x1 = x2; f1 = f2;
                    x2 = (1f - PhiRatio) * a + PhiRatio * b;
                    f2 = DistanceAtAngle(points, template, x2);
                }
            }
            return Mathf.Min(f1, f2);
        }

        private static float DistanceAtAngle(List<Vector2> points, IReadOnlyList<Vector2> template, float radians)
        {
            var rotated = RotateBy(points, radians);
            return PathDistance(rotated, template);
        }

        /// <summary>Coarse-samples the full circle to find the right basin, then refines with golden-
        /// section search — a plain wide golden-section search risks the wrong local minimum for
        /// symmetric shapes (e.g. a square has 4, 90deg apart).</summary>
        private static float FindBestRotationDistance(
            List<Vector2> points, IReadOnlyList<Vector2> template, int coarseSamples = 24)
        {
            float step = 2f * Mathf.PI / coarseSamples;
            float bestAngle = 0f;
            float bestCoarseDistance = float.MaxValue;

            for (int i = 0; i < coarseSamples; i++)
            {
                float angle = -Mathf.PI + i * step;
                float distance = DistanceAtAngle(points, template, angle);
                if (distance < bestCoarseDistance)
                {
                    bestCoarseDistance = distance;
                    bestAngle = angle;
                }
            }

            float halfWindow = step * 0.6f; // a bit over half a coarse step, to safely bracket the true minimum
            return DistanceAtBestAngle(points, template, bestAngle - halfWindow, bestAngle + halfWindow, AnglePrecision);
        }
    }
}
