#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
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
        private string hardNegativesFolder = "MLData/HardNegatives";
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
        private int negativeImagesCount = 600;
        private float maxSimilarityToRealSpell = 0.75f;
        private float maxAspectDifferenceToReject = 0.15f;
        private const int NegativeShapePointCount = 48;
        private const int MaxNegativeGenerationAttempts = 30;
        private const float MaxCornerDifferenceToReject = 0.1f;
        private const float NegativeJitterFraction = 0.08f;

        private bool regenerateAllClasses = true;
        private readonly HashSet<string> selectedClassesToRegenerate = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

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
            hardNegativesFolder = EditorGUILayout.TextField("Hard Negatives Folder", hardNegativesFolder);
            EditorGUILayout.HelpBox(
                "Images saved here (via HardNegativeCapture at runtime), under a subfolder per class " +
                "name, get folded into that class's train/val split on generate — on top of, not " +
                "instead of, the procedural/recorded sources.", MessageType.None);

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
                maxAspectDifferenceToReject = EditorGUILayout.Slider(
                    "Max Aspect Diff To Reject", maxAspectDifferenceToReject, 0.1f, 1f);
            }

            GUILayout.Space(8);
            useFixedSeed = EditorGUILayout.Toggle("Use Fixed Seed", useFixedSeed);
            using (new EditorGUI.DisabledScope(!useFixedSeed))
                seed = EditorGUILayout.IntField("Seed", seed);

            GUILayout.Space(8);
            GUILayout.Label("Selective Regeneration", EditorStyles.boldLabel);
            regenerateAllClasses = EditorGUILayout.Toggle("Regenerate All Classes", regenerateAllClasses);
            using (new EditorGUI.DisabledScope(regenerateAllClasses))
            {
                EditorGUILayout.HelpBox(
                    "Only checked classes are wiped and regenerated — everything else on disk (and in " +
                    "classes.json) is left exactly as it is. Handy when you only changed one class's " +
                    "templates or the negative-class generator, and don't want to wait on the rest.",
                    MessageType.None);

                foreach (string className in GetAvailableClassNames())
                {
                    bool isSelected = selectedClassesToRegenerate.Contains(className);
                    bool newValue = EditorGUILayout.ToggleLeft(className, isSelected);
                    if (newValue) selectedClassesToRegenerate.Add(className);
                    else selectedClassesToRegenerate.Remove(className);
                }
            }

            GUILayout.Space(12);
            if (GUILayout.Button("Generate Dataset", GUILayout.Height(32)))
            {
                Generate();
            }
        }

        /// <summary>Class names available for the selective-regeneration checkboxes: every distinct
        /// spellName found among recorded templates, plus the negative class if enabled. Cheap enough
        /// to recompute every OnGUI call (names only, no point data).</summary>
        private List<string> GetAvailableClassNames()
        {
            var names = AssetDatabase.FindAssets("t:SpellTemplate", new[] { templatesFolder })
                .Select(guid => AssetDatabase.LoadAssetAtPath<SpellTemplate>(AssetDatabase.GUIDToAssetPath(guid)))
                .Where(t => t != null)
                .Select(t => t.spellName)
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .OrderBy(n => n, StringComparer.OrdinalIgnoreCase)
                .ToList();
            if (includeNegativeClass) names.Add(negativeClassName);
            return names;
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
            string hardNegativesRoot = Path.Combine(projectRoot, hardNegativesFolder);

            List<IGrouping<string, SpellTemplate>> groups = templates
                .GroupBy(t => t.spellName, StringComparer.OrdinalIgnoreCase)
                .OrderBy(g => g.Key, StringComparer.OrdinalIgnoreCase)
                .ToList();

            bool ShouldRegenerate(string className) =>
                regenerateAllClasses || selectedClassesToRegenerate.Contains(className);

            if (regenerateAllClasses)
            {
                if (Directory.Exists(root))
                {
                    if (!EditorUtility.DisplayDialog("Overwrite Dataset?",
                            $"'{outputFolder}' already exists. Delete and regenerate?", "Delete & Regenerate", "Cancel"))
                        return;
                    Directory.Delete(root, true);
                }
            }
            else
            {
                // Selective mode: wipe only the classes being regenerated; everything else on disk stays.
                var classesToWipe = groups.Select(g => g.Key).Where(ShouldRegenerate)
                    .Concat(includeNegativeClass && ShouldRegenerate(negativeClassName)
                        ? new[] { negativeClassName } : Array.Empty<string>())
                    .ToList();

                if (classesToWipe.Count == 0)
                {
                    Debug.LogWarning("[SyntheticDatasetGenerator] No classes selected to regenerate.");
                    return;
                }

                foreach (string className in classesToWipe) DeleteClassFolders(trainRoot, valRoot, SanitizeClassName(className));
            }
            Directory.CreateDirectory(trainRoot);
            Directory.CreateDirectory(valRoot);

            System.Random rng = useFixedSeed ? new System.Random(seed) : new System.Random();
            int totalWritten = 0;
            bool cancelled = false;
            var recordingCounts = new List<string>();

            for (int g = 0; g < groups.Count && !cancelled; g++)
            {
                IGrouping<string, SpellTemplate> group = groups[g];
                if (!ShouldRegenerate(group.Key)) continue;
                string className = SanitizeClassName(group.Key);

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

                    // Round-robin, not random, so every recording gets an even share deterministically.
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

                int merged = MergeHardNegatives(hardNegativesRoot, className, trainDir, valDir, valSplit);
                if (merged > 0)
                {
                    totalWritten += merged;
                    recordingCounts[recordingCounts.Count - 1] += $" + {merged} hard negative{(merged == 1 ? "" : "s")}";
                }
            }

            bool shouldGenerateNegative = includeNegativeClass && !cancelled && ShouldRegenerate(negativeClassName);
            if (shouldGenerateNegative)
            {
                string className = SanitizeClassName(negativeClassName);

                string trainDir = Path.Combine(trainRoot, className);
                string valDir = Path.Combine(valRoot, className);
                Directory.CreateDirectory(trainDir);
                Directory.CreateDirectory(valDir);

                int valCount = Mathf.RoundToInt(negativeImagesCount * valSplit);

                // Phase 1: generate + reject-check in parallel (pure struct math, no Unity engine calls,
                // safe off the main thread). Each iteration gets its own System.Random — not thread-safe
                // to share one.
                var shapes = new List<Vector2>[negativeImagesCount];
                var thicknesses = new float[negativeImagesCount];
                var perImageRejections = new int[negativeImagesCount];
                int baseSeed = useFixedSeed ? seed : Environment.TickCount;

                EditorUtility.DisplayProgressBar("Generating Synthetic Dataset",
                    $"{negativeClassName}: searching for non-colliding shapes...", 0f);
                Parallel.For(0, negativeImagesCount, i =>
                {
                    var localRng = new System.Random(baseSeed + i);
                    int localRejections = 0;
                    shapes[i] = GenerateNegativeShape(
                        localRng, NegativeShapePointCount, templates, maxSimilarityToRealSpell,
                        maxAspectDifferenceToReject, ref localRejections);
                    thicknesses[i] = Lerp(localRng, strokeThicknessMin, strokeThicknessMax);
                    perImageRejections[i] = localRejections;
                });

                // Phase 2: rasterize + save, sequential (Texture2D/EncodeToPNG need the main thread).
                int rejectedAttempts = 0;
                for (int i = 0; i < negativeImagesCount; i++)
                {
                    if (EditorUtility.DisplayCancelableProgressBar("Generating Synthetic Dataset",
                            $"{negativeClassName}: saving {i + 1}/{negativeImagesCount}", (float)i / negativeImagesCount))
                    {
                        cancelled = true;
                        break;
                    }

                    rejectedAttempts += perImageRejections[i];
                    Texture2D tex = StrokeRasterizer.Rasterize(shapes[i], imageSize, thicknesses[i], 0.12f);

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

                int mergedNegatives = MergeHardNegatives(hardNegativesRoot, className, trainDir, valDir, valSplit);
                if (mergedNegatives > 0)
                {
                    totalWritten += mergedNegatives;
                    Debug.Log($"[SyntheticDatasetGenerator] Folded in {mergedNegatives} hard negative(s) for '{negativeClassName}'.");
                }
            }

            EditorUtility.ClearProgressBar();

            // Built from whatever class folders actually exist on disk, not just what this run
            // touched — correct for both full and selective regeneration.
            List<string> finalClassNames = Directory.Exists(trainRoot)
                ? Directory.GetDirectories(trainRoot).Select(Path.GetFileName)
                    .OrderBy(n => n, StringComparer.OrdinalIgnoreCase).ToList()
                : new List<string>();
            File.WriteAllText(Path.Combine(root, "classes.json"), ToJsonArray(finalClassNames));

            if (cancelled)
            {
                Debug.LogWarning($"Dataset generation cancelled after {totalWritten} images.");
                return;
            }

            if (shouldGenerateNegative) recordingCounts.Add($"{negativeClassName} (procedural)");
            Debug.Log($"Generated {totalWritten} images this run. {finalClassNames.Count} total classes now " +
                      $"on disk at: {root}\nRegenerated this run: {string.Join(", ", recordingCounts)}");
            EditorUtility.RevealInFinder(root);
        }

        private static void DeleteClassFolders(string trainRoot, string valRoot, string className)
        {
            string trainDir = Path.Combine(trainRoot, className);
            string valDir = Path.Combine(valRoot, className);
            if (Directory.Exists(trainDir)) Directory.Delete(trainDir, true);
            if (Directory.Exists(valDir)) Directory.Delete(valDir, true);
        }

        /// <summary>Copies (not moves — the staging folder is a persistent bank) whatever's in
        /// hardNegativesRoot/&lt;className&gt; into the class's train/val split.</summary>
        private static int MergeHardNegatives(string hardNegativesRoot, string className, string trainDir, string valDir, float valSplit)
        {
            string sourceDir = Path.Combine(hardNegativesRoot, className);
            if (!Directory.Exists(sourceDir)) return 0;

            string[] files = Directory.GetFiles(sourceDir, "*.png");
            if (files.Length == 0) return 0;

            int valCount = Mathf.RoundToInt(files.Length * valSplit);
            for (int i = 0; i < files.Length; i++)
            {
                string destDir = i < valCount ? valDir : trainDir;
                string destPath = Path.Combine(destDir, $"hardneg_{Path.GetFileName(files[i])}");
                File.Copy(files[i], destPath, overwrite: true);
            }
            return files.Length;
        }

        /// <summary>Rotation, then a non-uniform per-axis stretch (unlike uniform scale, this survives
        /// StrokeRasterizer's fit-to-square step), then per-point jitter scaled to the shape's own size.</summary>
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

        /// <summary>Picks a random "junk" archetype, jitters it, and retries (bounded) until
        /// TooCloseToAnyTemplate says it's genuinely dissimilar from every real spell.</summary>
        private static List<Vector2> GenerateNegativeShape(
            System.Random rng, int pointCount, IReadOnlyList<SpellTemplate> realTemplates,
            float maxSimilarity, float maxAspectDifference, ref int rejectedAttempts)
        {
            List<Vector2> candidate = null;
            for (int attempt = 0; attempt < MaxNegativeGenerationAttempts; attempt++)
            {
                candidate = rng.Next(7) switch
                {
                    0 => GenerateEllipse(rng, pointCount),
                    1 => GeneratePolygon(rng, pointCount),
                    2 => GenerateScribble(rng, pointCount),
                    3 => GenerateSpiky(rng, pointCount),
                    4 => GenerateChaoticSpiral(rng, pointCount),
                    5 => GenerateTangle(rng, pointCount),
                    _ => GenerateStarPolygon(rng, pointCount),
                };
                candidate = ApplyNaturalJitter(candidate, rng);

                if (!TooCloseToAnyTemplate(candidate, realTemplates, maxSimilarity, maxAspectDifference)) return candidate;

                rejectedAttempts++;
            }
            return candidate; // gave up after MaxNegativeGenerationAttempts — use the last try anyway
        }

        /// <summary>Two independent ways to be "too close to a real spell": $1 shape score + aspect
        /// match (catches structural collisions), OR aspect + corner-score match directly (a coarser
        /// but more robust fingerprint — $1's distance alone isn't reliable enough to gate on, since it
        /// can read artificially low for shapes that are structurally similar but not a precise
        /// point-for-point match).</summary>
        private static bool TooCloseToAnyTemplate(
            List<Vector2> candidate, IReadOnlyList<SpellTemplate> realTemplates,
            float maxSimilarity, float maxAspectDifference)
        {
            float candidateAspect = DollarOneRecognizer.ComputeAspectRatio(candidate);
            float candidateCornerScore = DollarOneRecognizer.ComputeCornerScore(DollarOneRecognizer.Normalize(candidate));

            foreach (SpellTemplate template in realTemplates)
            {
                if (template == null || template.CapturedPoints.Count < 3) continue;

                float templateAspect = DollarOneRecognizer.ComputeAspectRatio(template.CapturedPoints);
                float aspectDiff = Mathf.Abs(candidateAspect - templateAspect);
                if (aspectDiff >= maxAspectDifference) continue;

                float cornerDiff = Mathf.Abs(candidateCornerScore - template.CornerScore);
                if (cornerDiff < MaxCornerDifferenceToReject) return true;
            }

            return ShapeSimilarityToAnyTemplate(candidate, realTemplates) >= maxSimilarity
                   && AspectMatchesAnyTemplate(candidateAspect, realTemplates, maxAspectDifference);
        }

        private static bool AspectMatchesAnyTemplate(
            float candidateAspect, IReadOnlyList<SpellTemplate> realTemplates, float maxAspectDifference)
        {
            foreach (SpellTemplate template in realTemplates)
            {
                if (template == null || template.CapturedPoints.Count < 3) continue;
                float templateAspect = DollarOneRecognizer.ComputeAspectRatio(template.CapturedPoints);
                if (Mathf.Abs(candidateAspect - templateAspect) < maxAspectDifference) return true;
            }
            return false;
        }

        /// <summary>Small per-point jitter so negative shapes look hand-drawn, not mathematically
        /// perfect. Scaled per-point by that point's own distance from the centroid, not one global
        /// bounding-box size — a spiky shape's inner points sit much closer to center than its outer
        /// tips, so a single global scale would swamp the inner detail in noise.</summary>
        private static List<Vector2> ApplyNaturalJitter(List<Vector2> points, System.Random rng)
        {
            Vector2 centroid = Vector2.zero;
            foreach (Vector2 p in points) centroid += p;
            centroid /= points.Count;

            var result = new List<Vector2>(points.Count);
            foreach (Vector2 p in points)
            {
                float localScale = Mathf.Max(Vector2.Distance(p, centroid), 1f);
                Vector2 jitter = new Vector2(Lerp(rng, -NegativeJitterFraction, NegativeJitterFraction),
                                              Lerp(rng, -NegativeJitterFraction, NegativeJitterFraction)) * localScale;
                result.Add(p + jitter);
            }
            return result;
        }

        /// <summary>Checks both point orderings (our archetypes always trace one fixed direction, but
        /// a real template could be either) with the full 360deg rotation search (unlike live
        /// gameplay's +/-45deg window — a rotationally symmetric candidate has no meaningful "start
        /// point", so the narrow window would only sometimes land close enough to reveal true similarity).</summary>
        private static float ShapeSimilarityToAnyTemplate(List<Vector2> candidate, IReadOnlyList<SpellTemplate> realTemplates)
        {
            float forward = DollarOneRecognizer.Recognize(candidate, realTemplates, fullRotationSearch: true).score;

            var reversed = new List<Vector2>(candidate);
            reversed.Reverse();
            float backward = DollarOneRecognizer.Recognize(reversed, realTemplates, fullRotationSearch: true).score;

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

        /// <summary>Alternates outer/inner radius per vertex to produce spiky, concave, trident-like
        /// shapes — the only archetype with real concave structure. Irregular spacing/ratio keeps it
        /// from reliably looking like a clean Star (the $1 similarity check is the actual safety net
        /// for that).</summary>
        private static List<Vector2> GenerateSpiky(System.Random rng, int pointCount)
        {
            int prongCount = rng.Next(3, 8);
            var vertices = new List<Vector2>(prongCount * 2 + 1);

            for (int i = 0; i < prongCount * 2; i++)
            {
                bool isOuter = i % 2 == 0;
                float angle = (float)i / (prongCount * 2) * Mathf.PI * 2f + Lerp(rng, -0.15f, 0.15f);
                float outerRadius = Lerp(rng, 80f, 150f);
                float radius = isOuter ? outerRadius : outerRadius * Lerp(rng, 0.15f, 0.6f);
                vertices.Add(new Vector2(Mathf.Cos(angle) * radius, Mathf.Sin(angle) * radius));
            }
            vertices.Add(vertices[0]);
            return ResamplePolyline(vertices, pointCount);
        }

        /// <summary>A multi-turn spiral with sharp local radius spikes at random points along its
        /// length — protrusions scatter across many directions (unlike GenerateSpiky's evenly-spaced
        /// layout), denser and less symmetric than any other archetype.</summary>
        private static List<Vector2> GenerateChaoticSpiral(System.Random rng, int pointCount)
        {
            float turns = Lerp(rng, 1.3f, 3f);
            float totalAngle = turns * Mathf.PI * 2f;
            float startRadius = Lerp(rng, 10f, 40f);
            float endRadius = Lerp(rng, 80f, 160f);
            float rotationOffset = Lerp(rng, 0f, 360f) * Mathf.Deg2Rad;

            int protrusionCount = rng.Next(3, 8);
            var protrusionAngles = new float[protrusionCount];
            var protrusionStrengths = new float[protrusionCount];
            for (int p = 0; p < protrusionCount; p++)
            {
                protrusionAngles[p] = Lerp(rng, 0f, totalAngle);
                protrusionStrengths[p] = Lerp(rng, 0.3f, 0.9f);
            }
            const float protrusionWindow = 0.35f; // radians of spiral angle each spike affects

            var points = new List<Vector2>(pointCount);
            for (int i = 0; i < pointCount; i++)
            {
                float t = (float)i / (pointCount - 1);
                float angle = t * totalAngle;
                float radius = Mathf.Lerp(startRadius, endRadius, t);

                float bump = 0f;
                for (int p = 0; p < protrusionCount; p++)
                {
                    float diff = Mathf.Abs(angle - protrusionAngles[p]);
                    if (diff < protrusionWindow) bump += protrusionStrengths[p] * (1f - diff / protrusionWindow);
                }
                radius *= 1f + bump;

                float finalAngle = angle + rotationOffset;
                points.Add(new Vector2(Mathf.Cos(finalAngle) * radius, Mathf.Sin(finalAngle) * radius));
            }
            return points;
        }

        /// <summary>Several overlapping circular loops, each sweeping past 360deg — self-intersects
        /// both within a loop and across loops, unlike the single continuous curves of the other
        /// archetypes.</summary>
        private static List<Vector2> GenerateTangle(System.Random rng, int pointCount)
        {
            int loopCount = rng.Next(3, 6);
            int pointsPerLoop = Mathf.Max(pointCount / loopCount, 4);
            var points = new List<Vector2>(pointsPerLoop * loopCount);

            for (int i = 0; i < loopCount; i++)
            {
                Vector2 center = new Vector2(Lerp(rng, -40f, 40f), Lerp(rng, -40f, 40f));
                float radius = Lerp(rng, 40f, 90f);
                float startAngle = Lerp(rng, 0f, 360f) * Mathf.Deg2Rad;
                float sweep = Lerp(rng, 220f, 420f) * Mathf.Deg2Rad; // often > full circle: guarantees self-crossing

                for (int j = 0; j < pointsPerLoop; j++)
                {
                    float t = (float)j / (pointsPerLoop - 1);
                    float angle = startAngle + t * sweep;
                    points.Add(center + new Vector2(Mathf.Cos(angle), Mathf.Sin(angle)) * radius);
                }
            }
            return ResamplePolyline(points, pointCount);
        }

        /// <summary>Built like an actual star polygon: vertices around a circle, connected non-
        /// adjacently (skip &gt;= 2) instead of to neighbors — self-intersecting straight strokes,
        /// same principle as a pentagram.</summary>
        private static List<Vector2> GenerateStarPolygon(System.Random rng, int pointCount)
        {
            int vertexCount = rng.Next(4, 9);
            int skip = rng.Next(2, Mathf.Max(3, vertexCount - 1));
            float baseRadius = Lerp(rng, 80f, 150f);
            float rotationOffset = Lerp(rng, 0f, 360f) * Mathf.Deg2Rad;

            var vertices = new Vector2[vertexCount];
            for (int i = 0; i < vertexCount; i++)
            {
                float angle = (float)i / vertexCount * Mathf.PI * 2f + rotationOffset;
                float radius = baseRadius * Lerp(rng, 0.8f, 1.2f);
                vertices[i] = new Vector2(Mathf.Cos(angle) * radius, Mathf.Sin(angle) * radius);
            }

            var path = new List<Vector2>(vertexCount + 1);
            int idx = 0;
            for (int s = 0; s <= vertexCount; s++)
            {
                path.Add(vertices[idx % vertexCount]);
                idx += skip;
            }
            return ResamplePolyline(path, pointCount);
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
