import 'dart:io';

import 'package:flutter/material.dart';
import 'package:sayan_s_application11/presentation/start_page_screen/start_page_screen.dart';
import 'package:sayan_s_application11/presentation/scan_page_screen/scan_page_screen.dart';
import 'package:sayan_s_application11/presentation/micro_one_screen/micro_one_screen.dart';
import 'package:sayan_s_application11/presentation/micro_screen/micro_screen.dart';
import 'package:sayan_s_application11/presentation/app_navigation_screen/app_navigation_screen.dart';

class AppRoutes {
  static const String startPageScreen = '/start_page_screen';

  static const String scanPageScreen = '/scan_page_screen';

  static const String microOneScreen = '/micro_one_screen';

  static const String microScreen = '/micro_screen';

  static const String appNavigationScreen = '/app_navigation_screen';

  static Map<String, WidgetBuilder> routes = {
    startPageScreen: (context) => StartPageScreen(),
    scanPageScreen: (context) => ScanPageScreen(),
    microOneScreen: (context) =>
        MicroOneScreen(imageFile: File('path/to/image')),
    microScreen: (context) => MicroScreen(imageFile: File('path/to/image')),
    appNavigationScreen: (context) => AppNavigationScreen()
  };
}
