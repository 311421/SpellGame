using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.InputSystem;

namespace SpellDrawing.CV
{
    /// <summary>Dev-time tool: draw a gesture, see it misclassified, press captureKey to bank it as a
    /// hard negative for whichever class it wrongly matched. Saved to a staging folder that
    /// SyntheticDatasetGenerator folds into the training set on its next run.</summary>
    [RequireComponent(typeof(StrokeDrawer))]
    public class HardNegativeCapture : MonoBehaviour
    {
        [Tooltip("Which class this capture corrects. Usually the negative class, but works for any " +
                 "class name SyntheticDatasetGenerator recognizes.")]
        [SerializeField] private string targetClassName = "NotASpell";

        [SerializeField] private Key captureKey = Key.N;

        [Header("Rasterization")]
        [Tooltip("Should match what the model actually trains/infers at (SyntheticDatasetGenerator's " +
                 "Image Size / CnnSpellClassifier's settings).")]
        [SerializeField] private int imageSize = 48;
        [SerializeField] private float strokeThicknessPx = 1.75f;
        [SerializeField] private float padding = 0.12f;

        [SerializeField] private string hardNegativesFolder = "MLData/HardNegatives";

        private StrokeDrawer _drawer;
        private List<List<Vector2>> _lastGesture;

        private void Awake() => _drawer = GetComponent<StrokeDrawer>();
        private void OnEnable() => _drawer.OnGestureCompleted += HandleGestureCompleted;
        private void OnDisable() => _drawer.OnGestureCompleted -= HandleGestureCompleted;

        private void HandleGestureCompleted(List<List<Vector2>> strokes) => _lastGesture = strokes;

        private void Update()
        {
            Keyboard keyboard = Keyboard.current;
            if (keyboard == null) return;
            if (keyboard[captureKey].wasPressedThisFrame) CaptureLastGestureAsHardNegative();
        }

        private void CaptureLastGestureAsHardNegative()
        {
            if (_lastGesture == null || _lastGesture.Count == 0)
            {
                Debug.LogWarning("[HardNegativeCapture] No gesture drawn yet — draw one, see it " +
                                  "misclassified, then press the capture key.", this);
                return;
            }

            Texture2D tex = StrokeRasterizer.Rasterize(_lastGesture, imageSize, strokeThicknessPx, padding);

            string projectRoot = Path.GetDirectoryName(Application.dataPath);
            // Must match SyntheticDatasetGenerator's SanitizeClassName exactly.
            string sanitizedClassName = targetClassName;
            foreach (char c in Path.GetInvalidFileNameChars()) sanitizedClassName = sanitizedClassName.Replace(c, '_');
            sanitizedClassName = sanitizedClassName.Replace(' ', '_');

            string dir = Path.Combine(projectRoot, hardNegativesFolder, sanitizedClassName);
            Directory.CreateDirectory(dir);

            string path = Path.Combine(dir, $"{DateTime.Now:yyyyMMdd_HHmmss_fff}.png");
            File.WriteAllBytes(path, tex.EncodeToPNG());
            Destroy(tex);

            Debug.Log($"[HardNegativeCapture] Saved hard negative for '{targetClassName}' to {path}. " +
                      "Run the dataset generator again to fold it into training.", this);
        }
    }
}
