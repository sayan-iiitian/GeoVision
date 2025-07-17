// theme_helper.dart
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:sayan_s_application11/main.dart';
import '../../core/app_export.dart';

String _appTheme = "primary";

/// Helper class for managing themes and colors.
class ThemeHelper {
  // A map of custom color themes supported by the app
  Map<String, PrimaryColors> _supportedCustomColor = {
    'primary': PrimaryColors(),
  };

  // A map of color schemes supported by the app
  Map<String, ColorScheme> _supportedColorScheme = {
    'primary': ColorSchemes.primaryColorScheme,
  };

  /// Changes the app theme based on the provided theme key.
  void changeTheme(String themeKey) {
    if (_supportedCustomColor.containsKey(themeKey) &&
        _supportedColorScheme.containsKey(themeKey)) {
      _appTheme = themeKey;
      globalMessengerKey.currentState?.showSnackBar(
        SnackBar(
          content: Text('Theme changed to $themeKey'),
        ),
      );
    } else {
      throw Exception(
          "$themeKey is not found. Make sure you have added this theme class in JSON. Try running flutter pub run build_runner");
    }
  }

  /// Returns the primary colors for the current theme.
  PrimaryColors _getThemeColors() {
    if (!_supportedCustomColor.containsKey(_appTheme)) {
      throw Exception(
          "$_appTheme is not found. Make sure you have added this theme class in JSON. Try running flutter pub run build_runner");
    }
    return _supportedCustomColor[_appTheme] ?? PrimaryColors();
  }

  /// Returns the current theme data.
  ThemeData _getThemeData() {
    if (!_supportedColorScheme.containsKey(_appTheme)) {
      throw Exception(
          "$_appTheme is not found. Make sure you have added this theme class in JSON. Try running flutter pub run build_runner");
    }

    var colorScheme =
        _supportedColorScheme[_appTheme] ?? ColorSchemes.primaryColorScheme;
    return ThemeData(
      visualDensity: VisualDensity.standard,
      colorScheme: colorScheme,
      textTheme: TextThemes.textTheme(colorScheme),
      scaffoldBackgroundColor: colorScheme.onPrimary,
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: appTheme.black900.withOpacity(0.2),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(8),
          ),
          shadowColor: appTheme.black900.withOpacity(0.1),
          elevation: 2,
          visualDensity: const VisualDensity(
            vertical: -4,
            horizontal: -4,
          ),
          padding: EdgeInsets.zero,
        ),
      ),
    );
  }

  /// Returns the primary colors for the current theme.
  PrimaryColors themeColor() => _getThemeColors();

  /// Returns the current theme data.
  ThemeData themeData() => _getThemeData();
}

/// Class containing the supported text theme styles.
class TextThemes {
  static TextTheme textTheme(ColorScheme colorScheme) => TextTheme(
        displayMedium: TextStyle(
          color: colorScheme.onPrimary.withOpacity(1),
          fontSize: 48,
          fontFamily: 'Poppins',
          fontWeight: FontWeight.w600,
        ),
        displaySmall: TextStyle(
          color: colorScheme.onPrimary.withOpacity(1),
          fontSize: 36,
          fontFamily: 'Poppins',
          fontWeight: FontWeight.w500,
        ),
        labelLarge: TextStyle(
          color: colorScheme.onPrimary.withOpacity(1),
          fontSize: 12,
          fontFamily: 'Poppins',
          fontWeight: FontWeight.w500,
        ),
        titleMedium: TextStyle(
          color: colorScheme.onPrimary.withOpacity(1),
          fontSize: 18,
          fontFamily: 'Poppins',
          fontWeight: FontWeight.w500,
        ),
        titleSmall: TextStyle(
          color: colorScheme.onPrimary.withOpacity(1),
          fontSize: 15,
          fontFamily: 'Poppins',
          fontWeight: FontWeight.w500,
        ),
      );
}

/// Class containing the supported color schemes.
class ColorSchemes {
  static final primaryColorScheme = ColorScheme.light(
    // Primary colors
    primary: Color(0XFFC4B0FF),

    // On colors(text colors)
    onPrimary: Color(0XA5FFFFFF),
  );
}

/// Class containing custom colors for a primary theme.
class PrimaryColors {
  // Black
  Color get black900 => Color(0XFF000000);
}

PrimaryColors get appTheme => ThemeHelper().themeColor();
ThemeData get theme => ThemeHelper().themeData();
