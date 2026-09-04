using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;

namespace SpellDrawing.CV
{
    /// <summary>Debug/learning tool: rasterizes each completed gesture and shows it in a RawImage,
    /// optionally saving as PNG. Not part of the spell-casting pipeline itself.</summary>
    [RequireComponent(typeof(StrokeDrawer))]
    public class StrokeRasterizerPreview : MonoBehaviour
    {
        [Header("Rasterization")]
        [SerializeField] private int textureSize = 64;
        [SerializeField] private float strokeThicknessPx = 3f;
        [SerializeField, Range(0f, 0.3f)] private float padding = 0.12f;

        [Header("Preview")]
        [Tooltip("Shows the rasterized bitmap here right after each stroke. Assign a RawImage " +
                 "(Tools > Spell Drawing > Add Rasterizer Preview sets this up for you).")]
        [SerializeField] private RawImage previewImage;

        [Header("Dataset Export")]
        [Tooltip("When enabled, also saves each rasterized stroke as a PNG to persistentDataPath — " +
                 "useful later for building a training set.")]
        [SerializeField] private bool saveToDisk;
        [SerializeField] private string saveFolder = "RasterizedStrokes";

        private StrokeDrawer _drawer;
        private Texture2D _lastTexture;

        private void Awake() => _drawer = GetComponent<StrokeDrawer>();

        private void OnEnable() => _drawer.OnGestureCompleted += HandleGestureCompleted;
        private void OnDisable() => _drawer.OnGestureCompleted -= HandleGestureCompleted;

        private void HandleGestureCompleted(List<List<Vector2>> strokes)
        {
            if (_lastTexture != null) Destroy(_lastTexture);
            _lastTexture = StrokeRasterizer.Rasterize(strokes, textureSize, strokeThicknessPx, padding);

            if (previewImage != null) previewImage.texture = _lastTexture;
            if (saveToDisk) SaveToDisk(_lastTexture);
        }

        private void SaveToDisk(Texture2D texture)
        {
            string dir = Path.Combine(Application.persistentDataPath, saveFolder);
            if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);

            string path = Path.Combine(dir, $"stroke_{DateTime.Now:yyyyMMdd_HHmmss_fff}.png");
            File.WriteAllBytes(path, texture.EncodeToPNG());
            Debug.Log($"[StrokeRasterizerPreview] Saved {path}", this);
        }
    }
}
