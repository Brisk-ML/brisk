"""Unit test PlotSettings."""
import json

import pytest
import plotnine as pn
from unittest import mock

from brisk.theme import plot_settings


@pytest.mark.unit
class TestPlotSettings:
    def test_invalid_file_format(self):
        """Test that invalid file format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid file file_format"):
            plot_settings.PlotSettings(file_format="invalid")

    def test_invalid_theme_type(self):
        """Test that invalid theme type raises TypeError."""
        with pytest.raises(TypeError, match="theme must be a plotnine theme object"):
            plot_settings.PlotSettings(theme="not a theme")

    def test_theme_none_uses_default(self):
        """Test that theme=None uses the default brisk theme."""
        with mock.patch('brisk.theme.plot_settings.brisk_theme') as mock_brisk_theme:
            mock_theme = pn.theme_minimal()
            mock_brisk_theme.return_value = mock_theme

            settings = plot_settings.PlotSettings(theme=None)

            mock_brisk_theme.assert_called_once()
            assert settings.theme == mock_theme

    def test_theme_override_false_extends_default(self):
        """Test that override=False extends the default theme."""
        custom_theme = pn.theme(text=pn.element_text(size=14))

        with mock.patch('brisk.theme.plot_settings.brisk_theme') as mock_brisk_theme:
            mock_default = pn.theme_minimal()
            mock_brisk_theme.return_value = mock_default

            settings = plot_settings.PlotSettings(theme=custom_theme, override=False)

            mock_brisk_theme.assert_called_once()
            assert isinstance(settings.theme, pn.theme)

    def test_theme_override_true_replaces_default(self):
        """Test that override=True replaces the default theme entirely."""
        custom_theme = pn.theme_classic()

        with mock.patch('brisk.theme.plot_settings.brisk_theme') as mock_brisk_theme:
            settings = plot_settings.PlotSettings(theme=custom_theme, override=True)

            mock_brisk_theme.assert_not_called()
            assert settings.theme is custom_theme

    def test_export_params_is_serializable(self):
        """Test that export_params returns JSON-serializable data."""

        settings = plot_settings.PlotSettings()
        params = settings.export_params()

        json_str = json.dumps(params)
        assert isinstance(json_str, str)

        deserialized = json.loads(json_str)
        assert deserialized["file_io_settings"]["file_format"] == "png"

    def test_partial_color_override(self):
        """Test that only specified colors are overridden."""
        settings = plot_settings.PlotSettings(
            primary_color="#FF0000"
        )

        colors = settings.get_colors()
        assert colors["primary_color"] == "#FF0000"
        assert colors["secondary_color"] == "#00A878"
        assert colors["accent_color"] == "#DE6B48"

    def test_all_valid_file_formats(self):
        """Test that all valid file formats are accepted."""
        for file_format in plot_settings.PlotSettings.VALID_FORMATS:
            settings = plot_settings.PlotSettings(file_format=file_format)
            assert settings.file_io_settings["file_format"] == file_format
