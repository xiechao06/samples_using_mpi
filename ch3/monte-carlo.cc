#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <cxxopts.hpp>
#include <spdlog/spdlog.h>
#include <iostream>
#include <string>
#include <fmt/core.h>

namespace mpi = boost::mpi;

const auto MINIMUM_LOG_LEVEL = spdlog::level::trace;

void root(mpi::communicator world, double e)
{
    spdlog::trace("root: send e {}", e);
}

int main(int argc, const char **argv)
{
    spdlog::set_level(MINIMUM_LOG_LEVEL);
    mpi::environment env;
    mpi::communicator world;

    auto rank = world.rank();

    if (rank == 0)
    {

        if (world.size() <= 1)
        {
            fmt::print(stderr, "At least 2 processes should be used");
            world.abort(1);
            return 1;
        }
        cxxopts::Options options{"monte-carlo", "Calculate PI using monte-carlo algorithm"};
        auto add_option = options.add_options();
        add_option("h,help", "Show this help screen");
        add_option("e", "epsilon", cxxopts::value<double>()->default_value("1e-5"));

        auto result = options.parse(argc, argv);

        if (result.count("help"))
        {
            fmt::print(fmt::runtime(options.help()));
            world.abort(1);
            return 1;
        }
        auto e = result["e"].as<double>();
        root(world, e);
    }
    else
    {
    }

    return 0;
}