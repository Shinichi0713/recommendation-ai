import { bootstrapApplication } from '@angular/platform-browser';
import { appConfig } from './app/app.config';
import { AppComponent } from './app/app.component';
import { FormsModule } from '@angular/forms';
import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
bootstrapApplication(AppComponent, appConfig)
  .catch((err) => console.error(err));


@NgModule({
  declarations: [
    // AppComponent,
    // 他のコンポーネント
  ],
  imports: [
    FormsModule, // これを追加
    CommonModule, // これを追加
    AppComponent
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }