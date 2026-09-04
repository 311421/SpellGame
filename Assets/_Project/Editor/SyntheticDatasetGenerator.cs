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
    /// rotation bias) with rotation/jitter/thickness and rasterizes to an ImageFolder-style layout
    /// (Dataset/train|val/&lt;class&gt;/*.png) for torchvision. Translation/scale aren't augmented since
    /// StrokeRasterizer recenters and rescales anyway. Rotation range is per-template
    /// (SpellTemplate.maxTrainingRotationDeg).</summary>
    public class SyntheticDatasetGenerator : EditorWindow
    {
        private string templatesFolder = "Assets/_Project/Spells/Templates";
        private string outputFolder = "MLData/Dataset";
        private int imagesPerClass = 400;
        private float valSplit = 0.15f;
        private int imageSize = 48;
        private float pointJitter = 0.04f;
        private float strokeThicknessMin = 1.25f;
        private float strokeThicknessMax = 2f;
        private bool useFixedSeed = true;
        private int seed = 12345;

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
            strokeThicknessMin = EditorGUILayout.FloatField("Stroke Thickness Min", strokeThicknessMin);
            strokeThicknessMax = EditorGUILayout.FloatField("Stroke Thickness Max", strokeThicknessMax);

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

            for (int t = 0; t < templates.Count && !cancelled; t++)
            {
                SpellTemplate template = templates[t];
                string className = SanitizeClassName(template.spellName);
                classNames.Add(className);

                string trainDir = Path.Combine(trainRoot, className);
                string valDir = Path.Combine(valRoot, className);
                Directory.CreateDirectory(trainDir);
                Directory.CreateDirectory(valDir);

                var basePoints = new List<Vector2>(template.CapturedPoints);
                float shapeScale = BoundingBoxMaxDimension(basePoints);
                int valCount = Mathf.RoundToInt(imagesPerClass * valSplit);

                for (int i = 0; i < imagesPerClass; i++)
                {
                    if (EditorUtility.DisplayCancelableProgressBar("Generating Synthetic Dataset",
                            $"{template.spellName}: {i + 1}/{imagesPerClass}",
                            (t + (float)i / imagesPerClass) / templates.Count))
                    {
                        cancelled = true;
                        break;
                    }

                    List<Vector2> augmented = Augment(basePoints, rng, template.maxTrainingRotationDeg, shapeScale);
                    float thickness = Lerp(rng, strokeThicknessMin, strokeThicknessMax);
                    Texture2D tex = StrokeRasterizer.Rasterize(augmented, imageSize, thickness, 0.12f);

                    string dir = i < valCount ? valDir : trainDir;
                    string path = Path.Combine(dir, $"{className}_{i:D4}.png");
                    File.WriteAllBytes(path, tex.EncodeToPNG());
                    DestroyImmediate(tex);
                    totalWritten++;
                }
            }

            EditorUtility.ClearProgressBar();

            File.WriteAllText(Path.Combine(root, "classes.json"), ToJsonArray(classNames));

            if (cancelled)
            {
                Debug.LogWarning($"Dataset generation cancelled after {totalWritten} images.");
                return;
            }

            Debug.Log($"Generated {totalWritten} images across {classNames.Count} classes at: {root}\n" +
                      $"Classes (alphabetical, matches torchvision.ImageFolder's ordering): {string.Join(", ", classNames)}");
            EditorUtility.RevealInFinder(root);
        }

        /// <summary>Rotation + per-point jitter (scaled to the shape's own bounding box, since
        /// CapturedPoints aren't pre-scaled like $1's NormalizedPoints).</summary>
        private List<Vector2> Augment(List<Vector2> basePoints, System.Random rng, float rotationRangeDeg, float shapeScale)
        {
            float angle = Lerp(rng, -rotationRangeDeg, rotationRangeDeg) * Mathf.Deg2Rad;
            float cos = Mathf.Cos(angle), sin = Mathf.Sin(angle);

            var result = new List<Vector2>(basePoints.Count);
            foreach (Vector2 p in basePoints)
            {
                Vector2 rotated = new Vector2(p.x * cos - p.y * sin, p.x * sin + p.y * cos);
                Vector2 jitter = new Vector2(Lerp(rng, -pointJitter, pointJitter), Lerp(rng, -pointJitter, pointJitter)) * shapeScale;
                result.Add(rotated + jitter);
            }
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
