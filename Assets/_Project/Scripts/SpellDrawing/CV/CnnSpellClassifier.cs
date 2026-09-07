using System;
using System.Collections.Generic;
using UnityEngine;
using Unity.InferenceEngine;

namespace SpellDrawing.CV
{
    /// <summary>Runs the trained CNN (spell_classifier.onnx) over completed gestures and logs its
    /// prediction — a parallel diagnostic alongside SpellCaster's $1 result, not wired into actual
    /// casting yet. Rasterization size/thickness should match what SyntheticDatasetGenerator used to
    /// train the model.</summary>
    [RequireComponent(typeof(StrokeDrawer))]
    public class CnnSpellClassifier : MonoBehaviour
    {
        [SerializeField] private ModelAsset modelAsset;
        [SerializeField] private TextAsset classesJson;
        [SerializeField] private float strokeThicknessPx = 1.75f;
        [SerializeField] private float padding = 0.12f;
        [SerializeField] private bool logPredictions = true;

        [Tooltip("Top prediction must clear this confidence to count as accepted, rather than a low-" +
                 "confidence near-tie being reported as if it were a real match.")]
        [SerializeField, Range(0f, 1f)] private float minConfidenceToAccept = 0.6f;

        private Worker _worker;
        private string[] _classNames;
        private int _imageSize;
        private StrokeDrawer _drawer;

        [Serializable]
        private class ClassesFile
        {
            public string[] classes;
            public int image_size;
        }

        private void Awake()
        {
            _drawer = GetComponent<StrokeDrawer>();

            if (modelAsset == null || classesJson == null)
            {
                Debug.LogWarning("[CnnSpellClassifier] Assign modelAsset and classesJson to enable CNN predictions.", this);
                enabled = false;
                return;
            }

            var parsed = JsonUtility.FromJson<ClassesFile>(classesJson.text);
            _classNames = parsed.classes;
            _imageSize = parsed.image_size;

            Model runtimeModel = ModelLoader.Load(modelAsset);
            _worker = new Worker(runtimeModel, BackendType.CPU);
        }

        private void OnEnable() => _drawer.OnGestureCompleted += HandleGestureCompleted;
        private void OnDisable() => _drawer.OnGestureCompleted -= HandleGestureCompleted;

        private void OnDestroy() => _worker?.Dispose();

        private void HandleGestureCompleted(List<List<Vector2>> strokes)
        {
            var result = Classify(strokes);
            if (!logPredictions) return;

            var parts = new List<string>();
            for (int i = 0; i < result.ranked.Count && i < 3; i++)
                parts.Add($"{result.ranked[i].name}:{result.ranked[i].confidence:P0}");

            string verdict = result.accepted
                ? $"'{result.name}'"
                : $"'{result.name}' — below {minConfidenceToAccept:P0} threshold, treated as uncertain";
            Debug.Log($"[CnnSpellClassifier] {verdict} ({result.confidence:P0}) — {string.Join(", ", parts)}", this);
        }

        /// <summary>Classifies a completed gesture. Returns the top prediction, whether it cleared
        /// minConfidenceToAccept, and every class ranked by confidence (softmax over raw logits).</summary>
        public (string name, float confidence, bool accepted, List<(string name, float confidence)> ranked) Classify(
            List<List<Vector2>> strokes)
        {
            float[] pixels = StrokeRasterizer.RasterizeToTensorData(strokes, _imageSize, strokeThicknessPx, padding);
            using var input = new Tensor<float>(new TensorShape(1, 1, _imageSize, _imageSize), pixels);

            _worker.Schedule(input);
            var output = _worker.PeekOutput() as Tensor<float>; // owned by the worker — don't Dispose
            float[] probs = Softmax(output.DownloadToArray());

            var ranked = new List<(string name, float confidence)>();
            for (int i = 0; i < _classNames.Length; i++) ranked.Add((_classNames[i], probs[i]));
            ranked.Sort((a, b) => b.confidence.CompareTo(a.confidence));

            bool accepted = ranked[0].confidence >= minConfidenceToAccept;
            return (ranked[0].name, ranked[0].confidence, accepted, ranked);
        }

        private static float[] Softmax(float[] logits)
        {
            float max = float.MinValue;
            foreach (float v in logits) if (v > max) max = v;

            var exp = new float[logits.Length];
            float sum = 0f;
            for (int i = 0; i < logits.Length; i++)
            {
                exp[i] = Mathf.Exp(logits[i] - max);
                sum += exp[i];
            }
            for (int i = 0; i < exp.Length; i++) exp[i] /= sum;
            return exp;
        }
    }
}
