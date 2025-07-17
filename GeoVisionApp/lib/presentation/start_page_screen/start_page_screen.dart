import 'package:flutter/material.dart';
import 'package:sayan_s_application11/core/app_export.dart';
import 'package:sayan_s_application11/widgets/custom_elevated_button.dart';

class StartPageScreen extends StatelessWidget {
  const StartPageScreen({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    mediaQueryData = MediaQuery.of(context);
    return SafeArea(
        child: Scaffold(
            extendBody: true,
            extendBodyBehindAppBar: true,
            body: Container(
                width: mediaQueryData.size.width,
                height: mediaQueryData.size.height,
                decoration: BoxDecoration(
                    color: theme.colorScheme.onPrimary,
                    image: DecorationImage(
                        image: AssetImage(ImageConstant.imgStartPage),
                        fit: BoxFit.cover)),
                child: Container(
                    width: double.maxFinite,
                    padding:
                        EdgeInsets.symmetric(horizontal: 48.h, vertical: 74.v),
                    child: Column(
                        mainAxisSize: MainAxisSize.min,
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Spacer(),
                          SizedBox(
                              width: 332.h,
                              child: Text("Stone \nIdentifier".toUpperCase(),
                                  maxLines: 2,
                                  overflow: TextOverflow.ellipsis,
                                  textAlign: TextAlign.center,
                                  style: theme.textTheme.displayMedium!
                                      .copyWith(height: 1.24))),
                          SizedBox(height: 6.v),
                          Container(
                              width: 250.h,
                              margin: EdgeInsets.only(left: 38.h, right: 43.h),
                              child: Text(
                                  "Find stone by color, shape, \npicture or scan it.",
                                  maxLines: 2,
                                  overflow: TextOverflow.ellipsis,
                                  textAlign: TextAlign.center,
                                  style: CustomTextStyles.titleMedium_1
                                      .copyWith(height: 1.24))),
                          SizedBox(height: 78.v),
                          CustomElevatedButton(
                            height: 72.v,
                            text: "Get Started",
                            margin: EdgeInsets.only(left: 9.h, right: 3.h),
                            buttonStyle: CustomButtonStyles.fillPrimary,
                            buttonTextStyle: theme.textTheme.displaySmall!,
                            onPressed: () {
                              onTapGetStarted(context);
                            },
                          )
                        ])))));
  }

  /// Navigates to the scanPageScreen when the action is triggered.
  onTapGetStarted(BuildContext context) {
    Navigator.pushNamed(context, AppRoutes.scanPageScreen);
  }
}
