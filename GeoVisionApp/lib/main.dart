// main.dart
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:sayan_s_application11/theme/theme_helper.dart';
import 'package:sayan_s_application11/routes/app_routes.dart';

var globalMessengerKey = GlobalKey<ScaffoldMessengerState>();
void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
  ]);

  ///Please update theme as per your need if required.
  ThemeHelper().changeTheme('primary');
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: theme,
      title: 'sayan_s_application11',
      debugShowCheckedModeBanner: false,
      initialRoute: AppRoutes.startPageScreen,
      routes: AppRoutes.routes,
    );
  }
}
