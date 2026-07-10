package com.predictionmarkets.weather.repository;

import com.predictionmarkets.weather.models.WeatherComObservation;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

public interface WeatherComObservationRepository extends JpaRepository<WeatherComObservation, Long> {
  @Query("""
      select o
        from WeatherComObservation o
       where (:requestLocationId is null or lower(o.requestLocationId) = lower(:requestLocationId))
         and (:obsId is null or lower(o.obsId) = lower(:obsId))
         and (:fromValidTimeGmt is null or o.validTimeGmt >= :fromValidTimeGmt)
         and (:toValidTimeGmt is null or o.validTimeGmt <= :toValidTimeGmt)
       order by o.validTimeGmt desc
      """)
  Page<WeatherComObservation> search(
      @Param("requestLocationId") String requestLocationId,
      @Param("obsId") String obsId,
      @Param("fromValidTimeGmt") Long fromValidTimeGmt,
      @Param("toValidTimeGmt") Long toValidTimeGmt,
      Pageable pageable);

  long countByRequestLocationIdAndObsIdAndValidTimeGmt(String requestLocationId,
                                                        String obsId,
                                                        long validTimeGmt);
}

