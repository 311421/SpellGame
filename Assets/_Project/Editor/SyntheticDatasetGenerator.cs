#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;
using SpellDrawing.CV;

namespace SpellDrawing.EditorTools
{
    /// <summary>Generates a synthetic image dataset for training a CNN spell classifier: augments each
    /// SpellTemplate.CapturedPoints (not NormalizedPoints — that's $1-derotated and would bake in a
    /// rotation bias) with rotation/aspect-stretch/jitter/thickness, plus an optional procedurally-
    /// generated "not a spell" negative class, and rasterizes to an ImageFolder-style layout
    /// (Dataset/train|val/&lt;class&gt;/*.png) for torchvision. Rotation range is per-template
    /// (SpellTemplate.maxTrainingRotationDeg).</summary>
    public class SyntheticDatasetGenerator : EditorWindow
    {
        private string templatesFolder = "Assets/_Project/Spells/Templates";
        private string outputFolder = "MLData/Dataset";
        private int imagesPerClass = 400;
        private float valSplit = 0.15f;
        private int imageSize = 48;
        private float pointJitter = 0.04f;
        private float aspectJitter = 0.2f;
        private float strokeThicknessMin = 1.25f;
        private float strokeThicknessMax = 2f;
        private bool useFixedSeed = true;
        private int seed = 12345;

        private bool includeNegativeClass = true;
        private string negativeClassName = "NotASpell";
        private int negativeImagesCount = 800;
        private float maxSimilarityToRealSpell = 0.7f;
        private const int NegativeShapePointCount = 48;
        private const int MaxNegativeGenerationAttempts = 30;

        /// <summary>One recorded example contributing to a class — classes with multiple
        /// SpellTemplate assets sharing the same spellName pool all their recordings together.</summary>
        private struct TemplateSource
        {
            public List<Vector2> basePoints;
            public float shapeScale;
            public float maxRotationDeg;
        }

        [MenuItem("Tools/Spell Drawing/ML/Generate Synthetic Dataset")]
        public static void ShowWindow() => GetWindow<SyntheticDatasetGenerator>("Generate Dataset");

        private void OnGUI()
        {
            GUILayout.Label("Source", EditorStyles.boldLabel);
            templatesFolder = EditorGUILayout.TextField("Templates Folder", templatesFolder);

            GUILayout.Space(8);
            GUILayout.Label("Output", EditorStyles.boldLabel);
            outputFolder = EditorGUILayout.TextField("Output Folder (project-relative)", outputFolder);
            imageSize = EditorGUILayout.IntField("Image Size (px)", imageSize);
            imagesPerClass = EditorGUILayout.IntField("Images Per Class", imagesPerClass);
            valSplit = EditorGUILayout.Slider("Validation Split", valSplit, 0.05f, 0.4f);

            GUILayout.Space(8);
            GUILayout.Label("Augmentation", EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(
                "Rotation range is set per-spell now — edit 'Max Training Rotation Deg' on each " +
                "SpellTemplate asset (wide for orientation-agnostic shapes, narrow for orientation-" +
                "sensitive ones).", MessageType.None);
            pointJitter = EditorGUILayout.Slider("Point Jitter", pointJitter, 0f, 0.15f);
            aspectJitter = EditorGUILayout.Slider("Aspect Jitter (+/-%)", aspectJitter, 0f, 0.5f);
            strokeThicknessMin = EditorGUILayout.FloatField("Stroke Thickness Min", strokeThicknessMin);
            strokeThicknessMax = EditorGUILayout.FloatField("Stroke Thickness Max", strokeThicknessMax);

            GUILayout.Space(8);
            GUILayout.Label("Negative Class", EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(
                "Adds a 'not a spell' class made of random junk shapes (ellipses, polygons, scribbles) " +
                "so the classifier can reject ambiguous input instead of being forced to confidently " +
                "pick one of your real spells for everything.", MessageType.None);
            includeNegativeClass = EditorGUILayout.Toggle("Include Negative Class", includeNegativeClass);
            using (new EditorGUI.DisabledScope(!includeNegativeClass))
            {
                negativeClassName = EditorGUILayout.TextField("Class Name", negativeClassName);
                negativeImagesCount = EditorGUILayout.IntField("Images", negativeImagesCount);
                maxSimilarityToRealSpell = EditorGUILayout.Slider(
                    "Max Similarity To Real Spell", maxSimilarityToRealSpell, 0.3f, 0.95f);
            }

            GUILayout.Space(8);
            useFixedSeed = EditorGUILayout.Toggle("Use Fixed Seed", useFixedSeed);
            using (new EditorGUI.DisabledScope(!useFixedSeed))
                seed = EditorGUILayout.IntField("Seed", seed);

            GUILayout.Space(12);
            if (GUILayout.Button("Generate Dataset", GUILayout.Height(32)))
            {
                Generate();
            }
        }

        private void Generate()
        {
            List<SpellTemplate> allTemplates = AssetDatabase.FindAssets("t:SpellTemplate", new[] { templatesFolder })
                .Select(guid => AssetDatabase.LoadAssetAtPath<SpellTemplate>(AssetDatabase.GUIDToAssetPath(guid)))
                .Where(t => t != null)
                .OrderBy(t => t.spellName, StringComparer.OrdinalIgnoreCase)
                .ToList();

            if (allTemplates.Count == 0)
            {
                Debug.LogError($"No SpellTemplate assets found in '{templatesFolder}'.");
                return;
            }

            List<SpellTemplate> templates = allTemplates.Where(t => t.CapturedPoints.Count > 0).ToList();
            List<SpellTemplate> skipped = allTemplates.Where(t => t.CapturedPoints.Count == 0).ToList();
            if (skipped.Count > 0)
            {
                Debug.LogWarning(
                    $"Skipping {skipped.Count} template(s) with no CapturedPoints (recorded before that " +
                    $"field existed): {string.Join(", ", skipped.Select(t => t.spellName))}. " +
                    "Re-record them (redraw + 'Save Last Gesture As Template') to include them.");
            }

            if (templates.Count == 0)
            {
                Debug.LogError("No templates with CapturedPoints to generate from — re-record your templates first.");
                return;
            }

            string projectRoot = Path.GetDirectoryName(Application.dataPath);
            string root = Path.Combine(projectRoot, outputFolder);
            string trainRoot = Path.Combine(root, "train");
            string valRoot = Path.Combine(root, "val");

            if (Directory.Exists(root))
            {
                if (!EditorUtility.DisplayDialog("Overwrite Dataset?",
                        $"'{outputFolder}' already exists. Delete and regenerate?", "Delete & Regenerate", "Cancel"))
                    return;
                Directory.Delete(root, true);
            }

            System.Random rng = useFixedSeed ? new System.Random(seed) : new System.Random();
            var classNames = new List<string>();
            int totalWritten = 0;
            bool cancelled = false;

            List<IGrouping<string, SpellTemplate>> groups = templates
                .GroupBy(t => t.spellName, StringComparer.OrdinalIgnoreCase)
                .OrderBy(g => g.Key, StringComparer.OrdinalIgnoreCase)
                .ToList();

            var recordingCounts = new List<string>();

            for (int g = 0; g < groups.Count && !cancelled; g++)
            {
                IGrouping<string, SpellTemplate> group = groups[g];
                string className = SanitizeClassName(group.Key);
                classNames.Add(className);

                List<TemplateSource> sources = group.Select(t =>
                {
                    var points = new List<Vector2>(t.CapturedPoints);
                    return new TemplateSource
                    {
                        basePoints = points,
                        shapeScale = BoundingBoxMaxDimension(points),
                        maxRotationDeg = t.maxTrainingRotationDeg,
                    };
                }).ToList();
                recordingCounts.Add($"{group.Key} ({sources.Count} recording{(sources.Count == 1 ? "" : "s")})");

                string trainDir = Path.Combine(trainRoot, className);
                string valDir = Path.Combine(valRoot, className);
                Directory.CreateDirectory(trainDir);
                Directory.CreateDirectory(valDir);

                int valCount = Mathf.RoundToInt(imagesPerClass * valSplit);

                for (int i = 0; i < imagesPerClass; i++)
                {
                    if (EditorUtility.DisplayCancelableProgressBar("Generating Synthetic Dataset",
                            $"{group.Key} ({sources.Count} recording{(sources.Count == 1 ? "" : "s")}): {i + 1}/{imagesPerClass}",
                            (g + (float)i / imagesPerClass) / groups.Count))
                    {
                        cancelled = true;
                        break;
                    }

                    // Cycle through recordings round-robin (not random) so every recording gets an
                    // even share of images regardless of how many there are, deterministically.
                    TemplateSource source = sources[i % sources.Count];
                    List<Vector2> augmented = Augment(source.basePoints, rng, source.maxRotationDeg, source.shapeScale);
                    float thickness = Lerp(rng, strokeThicknessMin, strokeThicknessMax);
                    Texture2D tex = StrokeRasterizer.Rasterize(augmented, imageSize, thickness, 0.12f);

                    string dir = i < valCount ? valDir : trainDir;
                    string path = Path.Combine(dir, $"{className}_{i:D4}.png");
                    File.WriteAllBytes(path, tex.EncodeToPNG());
                    DestroyImmediate(tex);
                    totalWritten++;
                }
            }

            if (includeNegativeClass && !cancelled)
            {
                string className = SanitizeClassName(negativeClassName);
                classNames.Add(className);

                string trainDir = Path.Combine(trainRoot, className);
                string valDir = Path.Combine(valRoot, className);
                Directory.CreateDirectory(trainDir);
                Directory.CreateDirectory(valDir);

                int valCount = Mathf.RoundToInt(negativeImagesCount * valSplit);
                int rejectedAttempts = 0;
                for (int i = 0; i < negativeImagesCount; i++)
                {
                    if (EditorUtility.DisplayCancelableProgressBar("Generating Synthetic Dataset",
                            $"{negativeClassName}: {i + 1}/{negativeImagesCount}", (float)i / negativeImagesCount))
                    {
                        cancelled = true;
                        break;
                    }

                    List<Vector2> shape = GenerateNegativeShape(
                        rng, NegativeShapePointCount, templates, maxSimilarityToRealSpell, ref rejectedAttempts);
                    float thickness = Lerp(rng, strokeThicknessMin, strokeThicknessMax);
                    Texture2D tex = StrokeRasterizer.Rasterize(shape, imageSize, thickness, 0.12f);

                    string dir = i < valCount ? valDir : trainDir;
                    string path = Path.Combine(dir, $"{className}_{i:D4}.png");
                    File.WriteAllBytes(path, tex.EncodeToPNG());
                    DestroyImmediate(tex);
                    totalWritten++;
                }

                if (rejectedAttempts > 0)
                {
                    Debug.Log($"[SyntheticDatasetGenerator] Discarded {rejectedAttempts} negative-class candidate(s) " +
                              $"that resembled a real spell too closely (score >= {maxSimilarityToRealSpell:F2}).");
                }
            }

            EditorUtility.ClearProgressBar();

            classNames.Sort(StringComparer.OrdinalIgnoreCase);
            File.WriteAllText(Path.Combine(root, "classes.json"), ToJsonArray(classNames));

            if (cancelled)
            {
                Debug.LogWarning($"Dataset generation cancelled after {totalWritten} images.");
                return;
            }

            if (includeNegativeClass) recordingCounts.Add($"{negativeClassName} (procedural)");
            Debug.Log($"Generated {totalWritten} images across {classNames.Count} classes at: {root}\n" +
                      $"Classes: {string.Join(", ", recordingCounts)}");
            EditorUtility.RevealInFinder(root);
        }

        /// <summary>Rotation, then a random per-axis stretch, then per-point jitter (jitter scaled to
        /// the shape's own bounding box, since CapturedPoints aren't pre-scaled like $1's
        /// NormalizedPoints). The stretch is non-uniform (independent X/Y factors) deliberately —
        /// unlike uniform scale, it survives StrokeRasterizer's fit-to-square step, so it's the only
        /// way to teach the classifier that a shape drawn a bit taller/wider than usual is still the
        /// same spell, rather than letting aspect ratio become a shortcut for telling classes apart.</summary>
        private List<Vector2> Augment(List<Vector2> basePoints, System.Random rng, float rotationRangeDeg, float shapeScale)
        {
            float angle = Lerp(rng, -rotationRangeDeg, rotationRangeDeg) * Mathf.Deg2Rad;
            float cos = Mathf.Cos(angle), sin = Mathf.Sin(angle);
            float stretchX = 1f + Lerp(rng, -aspectJitter, aspectJitter);
            float stretchY = 1f + Lerp(rng, -aspectJitter, aspectJitter);

            var result = new List<Vector2>(basePoints.Count);
            foreach (Vector2 p in basePoints)
            {
                Vector2 rotated = new Vector2(p.x * cos - p.y * sin, p.x * sin + p.y * cos);
                Vector2 stretched = new Vector2(rotated.x * stretchX, rotated.y * stretchY);
                Vector2 jitter = new Vector2(Lerp(rng, -pointJitter, pointJitter), Lerp(rng, -pointJitter, pointJitter)) * shapeScale;
                result.Add(stretched + jitter);
            }
            return result;
        }

        /// <summary>Picks one of a few "junk" archetypes (ellipse, irregular polygon, random-walk
        /// scribble), then checks it against the real templates via $1 — a near-circular ellipse or
        /// near-square quadrilateral can coincidentally land close to a real Circle/Square, and
        /// labeling that "not a spell" would directly contradict the real class's own training images
        /// of the same shape. Retries (bounded) until it finds one that's genuinely dissimilar.</summary>
        private static List<Vector2> GenerateNegativeShape(
            System.Random rng, int pointCount, IReadOnlyList<SpellTemplate> realTemplates,
            float maxSimilarity, ref int rejectedAttempts)
        {
            List<Vector2> candidate = null;
            for (int attempt = 0; attempt < MaxNegativeGenerationAttempts; attempt++)
            {
                candidate = rng.Next(3) switch
                {
                    0 => GenerateEllipse(rng, pointCount),
                    1 => GeneratePolygon(rng, pointCount),
                    _ => GenerateScribble(rng, pointCount),
                };

                if (SimilarityToAnyTemplate(candidate, realTemplates) < maxSimilarity) return candidate;

                rejectedAttempts++;
            }
            return candidate; // gave up after MaxNegativeGenerationAttempts — use the last try anyway
        }

        /// <summary>$1 only searches a rotation window, not tracing direction — a shape traced
        /// clockwise vs. the same shape traced counterclockwise can score as very dissimilar even
        /// though they'd rasterize identically. Our procedural shapes always trace one fixed
        /// direction (increasing angle), but a real recorded template could be either, so checking
        /// only the forward order would let same-direction-only collisions slip past. Checking both
        /// orderings and taking the higher score closes that gap.</summary>
        private static float SimilarityToAnyTemplate(List<Vector2> candidate, IReadOnlyList<SpellTemplate> realTemplates)
        {
            float forward = DollarOneRecognizer.Recognize(candidate, realTemplates).score;

            var reversed = new List<Vector2>(candidate);
            reversed.Reverse();
            float backward = DollarOneRecognizer.Recognize(reversed, realTemplates).score;

            return Mathf.Max(forward, backward);
        }

        private static List<Vector2> GenerateEllipse(System.Random rng, int pointCount)
        {
            float radiusX = Lerp(rng, 40f, 150f);
            float radiusY = Lerp(rng, 40f, 150f);
            float rotation = Lerp(rng, 0f, 360f) * Mathf.Deg2Rad;
            float cos = Mathf.Cos(rotation), sin = Mathf.Sin(rotation);

            var points = new List<Vector2>(pointCount);
            for (int i = 0; i < pointCount; i++)
            {
                float t = (float)i / (pointCount - 1) * Mathf.PI * 2f;
                float x = Mathf.Cos(t) * radiusX;
                float y = Mathf.Sin(t) * radiusY;
                points.Add(new Vector2(x * cos - y * sin, x * sin + y * cos));
            }
            return points;
        }

        private static List<Vector2> GeneratePolygon(System.Random rng, int pointCount)
        {
            int vertexCount = rng.Next(3, 9);
            var vertices = new List<Vector2>(vertexCount + 1);
            for (int i = 0; i < vertexCount; i++)
            {
                float angle = (float)i / vertexCount * Mathf.PI * 2f + Lerp(rng, -0.3f, 0.3f);
                float radius = Lerp(rng, 50f, 150f);
                vertices.Add(new Vector2(Mathf.Cos(angle) * radius, Mathf.Sin(angle) * radius));
            }
            vertices.Add(vertices[0]);
            return ResamplePolyline(vertices, pointCount);
        }

        private static List<Vector2> GenerateScribble(System.Random rng, int pointCount)
        {
            int controlCount = rng.Next(3, 7);
            var controls = new List<Vector2>(controlCount);
            Vector2 pos = Vector2.zero;
            controls.Add(pos);
            for (int i = 1; i < controlCount; i++)
            {
                float angle = Lerp(rng, 0f, 360f) * Mathf.Deg2Rad;
                float dist = Lerp(rng, 40f, 120f);
                pos += new Vector2(Mathf.Cos(angle) * dist, Mathf.Sin(angle) * dist);
                controls.Add(pos);
            }
            return ResamplePolyline(controls, pointCount);
        }

        /// <summary>Evenly resamples a polyline to exactly pointCount points — a local, minimal
        /// version of $1's own Resample, just for shaping negative-class junk shapes.</summary>
        private static List<Vector2> ResamplePolyline(List<Vector2> vertices, int pointCount)
        {
            float totalLength = 0f;
            for (int i = 1; i < vertices.Count; i++) totalLength += Vector2.Distance(vertices[i - 1], vertices[i]);
            if (totalLength < 0.0001f) return vertices;

            float interval = totalLength / (pointCount - 1);
            var result = new List<Vector2> { vertices[0] };
            float accumulated = 0f;
            Vector2 current = vertices[0];
            int vi = 0;

            while (result.Count < pointCount && vi < vertices.Count - 1)
            {
                Vector2 next = vertices[vi + 1];
                float segLen = Vector2.Distance(current, next);
                if (segLen > 0f && accumulated + segLen >= interval)
                {
                    float t = (interval - accumulated) / segLen;
                    Vector2 q = Vector2.Lerp(current, next, t);
                    result.Add(q);
                    current = q;
                    accumulated = 0f;
                }
                else
                {
                    accumulated += segLen;
                    current = next;
                    vi++;
                }
            }
            while (result.Count < pointCount) result.Add(vertices[vertices.Count - 1]);
            return result;
        }

        private static float BoundingBoxMaxDimension(List<Vector2> points)
        {
            Vector2 min = points[0], max = points[0];
            foreach (Vector2 p in points)
            {
                min = Vector2.Min(min, p);
                max = Vector2.Max(max, p);
            }
            return Mathf.Max(max.x - min.x, max.y - min.y, 0.0001f);
        }

        private static float Lerp(System.Random rng, float min, float max) => min + (float)rng.NextDouble() * (max - min);

        private static string SanitizeClassName(string name)
        {
            foreach (char c in Path.GetInvalidFileNameChars()) name = name.Replace(c, '_');
            return name.Replace(' ', '_');
        }

        private static string ToJsonArray(List<string> items) => "[" + string.Join(",", items.Select(s => $"\"{s}\"")) + "]";
    }
}
#endif
