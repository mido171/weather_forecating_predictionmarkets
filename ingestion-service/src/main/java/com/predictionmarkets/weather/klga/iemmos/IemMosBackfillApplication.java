package com.predictionmarkets.weather.klga.iemmos;

import com.predictionmarkets.weather.iem.IemProperties;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;

@SpringBootApplication(scanBasePackageClasses = {
    IemMosBackfillApplication.class
})
@EnableConfigurationProperties({ IemMosBackfillProperties.class, IemProperties.class })
public class IemMosBackfillApplication {
  public static void main(String[] args) {
    SpringApplication.run(IemMosBackfillApplication.class, args);
  }
}
