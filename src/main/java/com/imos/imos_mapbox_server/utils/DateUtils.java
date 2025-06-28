package com.imos.imos_mapbox_server.utils;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class DateUtils {
    /**
     * Generates a list of formatted date strings for the last seven days,
     * ending three days before today.
     *
     * <p>The list will contain dates in "yy-MM-dd" format, ordered from oldest to newest.</p>
     *
     * <p>For example, if today is 2025-04-28, the method will generate dates
     * from 2025-04-19 to 2025-04-25 (excluding the last three days: 26th, 27th, 28th).</p>
     *
     * @return a {@link List} of 7 date strings, each formatted as "yyyy-MM-dd"
     */
    public static List<String> getLastSevenDays() {
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd");
        LocalDate endDate = LocalDate.now().minusDays(3);

        return IntStream.rangeClosed(0, 6)
                .mapToObj(i -> endDate.minusDays(6 - i))
                .map(date -> date.format(formatter))
                .collect(Collectors.toList());
    }


    public static List<String> getCurrentMonthsInCurrentYear() {
        List<String> months = new ArrayList<>();
        LocalDate now = LocalDate.now();
        int currentYear = now.getYear();
        int currentMonth = now.getMonthValue();

        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd");

        for (int month = 1; month <= currentMonth; month++) {
            LocalDate date = LocalDate.of(currentYear, month, 1);
            months.add(date.format(formatter));
        }

        return months;
    }
}
